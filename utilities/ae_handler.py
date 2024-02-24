import os.path

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randn
from numpy.linalg import norm
import persim
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import pairwise_distances

from utilities.directoy_helper import DirectoryHelper
from utilities.tda_helper import TDAHelper


class AE(nn.Module):
    def __init__(self, encoderLayers, decoderLayers, activation=F.relu):
        super(AE, self).__init__()

        self.activation = activation

        self.enc1 = nn.Linear(encoderLayers[0], encoderLayers[1])
        self.enc2 = nn.Linear(encoderLayers[1], encoderLayers[2])
        self.enc3 = nn.Linear(encoderLayers[2], encoderLayers[3])

        self.dec1 = nn.Linear(decoderLayers[0], decoderLayers[1])
        self.dec2 = nn.Linear(decoderLayers[1], decoderLayers[2])
        self.dec3 = nn.Linear(decoderLayers[2], decoderLayers[3])
        self.criterion = nn.MSELoss()
        self.NOISE_STDDEV = 0.005
        self.determine_device()

    def forward(self, x):
        latentEnc1 = self.activation(self.enc1(x))
        latentEnc2 = self.activation(self.enc2(latentEnc1))
        code = self.activation(self.enc3(latentEnc2))

        latentDec1 = self.activation(self.dec1(code))
        latentDec2 = self.activation(self.dec2(latentDec1))
        recon = self.dec3(latentDec2)

        return recon, code, (latentEnc1, latentEnc2, latentDec1, latentDec2)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def add_gaussian_noise(self, input_data, noise_stddev, numpy=True):
        if numpy:
            return input_data + noise_stddev * randn(*input_data.shape)
        else:
            return input_data + noise_stddev * torch.randn(input_data.shape, device=self.device)

    def denoising_loss(self, data):
        noisy_data = self.add_gaussian_noise(data, self.NOISE_STDDEV, numpy=False)
        noisy_data = noisy_data.to(self.device)
        reconstructions, codes, latentReps = self(noisy_data)
        loss = self.criterion(reconstructions, data)
        return loss

    def vanilla_loss(self, data):
        reconstructions, codes, latentReps = self(data)
        loss = self.criterion(reconstructions, data)
        return loss

    def sparse_loss(self, base_criterion, data):
        pass

    def determine_device(self):
        self.train_on_gpu = torch.cuda.is_available()

        if not self.train_on_gpu:
            print('No GPU, training on CPU')
            self.device = torch.device('cpu')
        else:
            print('GPU found, training on GPU')
            self.device = torch.device('cuda')

    def collect_reps_data(self, loader, nSamples,
                          encReps, decReps,
                          codeLen, extrinsicDim,
                          denoising=False):

        cnt = 0

        AEcodes = np.zeros((nSamples, codeLen))
        AErecons = np.zeros((nSamples, extrinsicDim))

        encReps1 = np.zeros((nSamples, encReps[1]))
        encReps2 = np.zeros((nSamples, encReps[2]))

        decReps1 = np.zeros((nSamples, decReps[1]))
        decReps2 = np.zeros((nSamples, decReps[2]))

        loaderLabels = np.zeros(nSamples)

        self.eval()

        ll = 0.0

        for data, target in loader:

            if self.train_on_gpu:
                data, target = data.float().cuda(), target.float().cuda()
            else:
                data, target = data.float(), target.float()

            if denoising:
                data = self.add_gaussian_noise(data, self.NOISE_STDDEV, numpy=False)

            recons, codes, latentReps = self(data)
            losses = self.criterion(recons, data)

            ll += losses.item() * data.size(0)

            for ii in range(len(data)):
                AEcodes[cnt] = codes[ii].detach().cpu().numpy()
                AErecons[cnt] = recons[ii].detach().cpu().numpy()

                encReps1[cnt] = latentReps[0][ii].detach().cpu().numpy()
                encReps2[cnt] = latentReps[1][ii].detach().cpu().numpy()

                decReps1[cnt] = latentReps[2][ii].detach().cpu().numpy()
                decReps2[cnt] = latentReps[3][ii].detach().cpu().numpy()

                loaderLabels[cnt] = target[ii]

                cnt += 1

        print(ll / len(loader))

        return AErecons, AEcodes, encReps1, encReps2, decReps1, decReps2, loaderLabels

    def my_train(self, train_loader, n_epochs, file, compute_loss_fct=None, eta=0.001):
        if compute_loss_fct is None:
            compute_loss_fct = AE.vanilla_loss

        train_loss_min = np.inf

        # optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=eta)

        for epoch in range(1, n_epochs + 1):
            # monitor training loss
            train_loss = 0.0

            ###################
            # train the model #
            ###################
            for data, target in train_loader:

                if self.train_on_gpu:
                    data, target = data.float().cuda(), target.float().cuda()
                else:
                    data, target = data.float(), target.float()

                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                #             reconstructions, codes, latentReps = model(data)

                # calculate the loss
                #             loss = criterion(reconstructions, data)
                loss = compute_loss_fct(self,data)

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # perform a single optimization step (parameter update)
                optimizer.step()

                # update running training loss
                train_loss += loss.item() * data.size(0)

            # print avg training statistics
            train_loss = train_loss / len(train_loader)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                torch.save(self.state_dict(), file)

class DataController:
    def __init__(self, output_path=None):
        self.nPts = 2000  ## number of points sampled from each hypershpere
        self.intrinsicDim = 2  ## n-dim sphere
        self.extrinsicDim = 100
        self.nSpheres = 2  ## number of hypersheres
        self.radii = np.array([1, 2])  ##radii of the hyperspheres
        self.output_path = output_path if output_path is not None else "."

    def execute(self):
        np.random.seed(0)
        torch.manual_seed(0)
        A = randn(self.intrinsicDim + 1, self.extrinsicDim)
        # Generate data. All spheres are centerd at the origin
        self.Data = np.zeros((self.nSpheres * self.nPts, self.extrinsicDim))
        self.labels = np.zeros((self.nPts * self.nSpheres))

        for ii in range(self.nSpheres):
            foo = np.random.randn(self.nPts, self.intrinsicDim + 1)
            foo = foo @ A
            self.Data[ii * self.nPts:(ii + 1) * self.nPts] = self.radii[ii] * (foo / norm(foo, axis=1, keepdims=True))
            self.labels[ii * self.nPts:(ii + 1) * self.nPts] = ii
        self.visualize_data(self.Data)
        self.configure_data()

    def configure_data(self):
        self.dataTrain = Variable(torch.from_numpy(self.Data[::2])).requires_grad_(True)
        self.labelsTrain = Variable(torch.from_numpy(self.labels[::2])).requires_grad_(True)

        self.dataTest = Variable(torch.from_numpy(self.Data[1::2])).requires_grad_(True)
        self.labelsTest = Variable(torch.from_numpy(self.labels[1::2])).requires_grad_(True)

        ## create dataset and dataloader
        self.tensorTrainData = TensorDataset(self.dataTrain, self.labelsTrain)
        self.tensorTestData = TensorDataset(self.dataTest, self.labelsTest)

        bs = 100 ## batch size

        self.train_loader = DataLoader(self.tensorTrainData, batch_size=bs, shuffle=True)
        self.test_loader = DataLoader(self.tensorTestData, batch_size=bs, shuffle=False)

    def visualize_data(self, data):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        alphas = [1, 0.3]

        for ii in range(self.nSpheres):
            ax.scatter(data[ii * self.nPts:(ii + 1) * self.nPts, 0], data[ii * self.nPts:(ii + 1) * self.nPts, 1],
                       data[ii * self.nPts:(ii + 1) * self.nPts, 2], alpha=alphas[ii])

        plt.savefig(f"{self.output_path}/noisy_data_spheres.png")
        plt.close("all")

class ActivationFunctionTypes:
    RELU, LRELU, TANH = range(3)

    @staticmethod
    def get_all_afts():
        return [ActivationFunctionTypes.RELU,
                ActivationFunctionTypes.LRELU,
                ActivationFunctionTypes.TANH]

    @staticmethod
    def get_name(aft):
        if aft == ActivationFunctionTypes.RELU:
            return "AE_relu"

        if aft == ActivationFunctionTypes.LRELU:
            return "AE_lrelu"

        if aft == ActivationFunctionTypes.TANH:
            return "AE_tanh"

        return "NONE"

    @staticmethod
    def get_unprefixed_name(aft):
        if aft == ActivationFunctionTypes.RELU:
            return "RELU"

        if aft == ActivationFunctionTypes.LRELU:
            return "LRELU"

        if aft == ActivationFunctionTypes.TANH:
            return "TANH"

        return "NONE"

    @staticmethod
    def get_aft_path(output_path, aft):
        # look for the corresponding pretrained model *.pt
        aft_name = ActivationFunctionTypes.get_name(aft)
        aft_path = f"{output_path}/{aft_name}_data/"

        return os.path.normpath(aft_path)

    @staticmethod
    def get_aft_pretrained_model_filename(output_path, aft):
        aft_path = ActivationFunctionTypes.get_aft_path(output_path, aft)

        if not os.path.isdir(aft_path):
            return None

        pre_file = f"{aft_path}/{ActivationFunctionTypes.get_name(aft)}.pt"
        if not os.path.isfile(pre_file):
            return None

        return pre_file

class AETrainerController:
    def __init__(self, nepoch = 100, output_path="results", max_dim = 3, metric='manhattan'):
        self.data_ctrl = DataController(output_path=output_path)
        self.output_path = output_path
        self.current_path = output_path
        self.nepoch = nepoch
        self.max_dim = max_dim
        self.metric = metric
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

    def execute(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.data_ctrl.execute()
        self.encoderDims = [self.data_ctrl.extrinsicDim, 20, 10, self.data_ctrl.intrinsicDim + 1]
        self.decoderDims = self.encoderDims[::-1]

        self.train_relu()
        self.train_leaky_relu()
        self.train_tanh()

    def get_result_folder(self):
        return self.output_path

    def get_generated_folders(self):
        gen_folders = {aft: None for aft in ActivationFunctionTypes.get_all_afts()}
        folders = DirectoryHelper.get_all_subfolders(root_path=self.output_path,
                                                     dir_pattern="AE_")
        for aft in gen_folders:
            aft_folder = ActivationFunctionTypes.get_aft_path(self.output_path, aft)
            if aft_folder in folders:
                gen_folders[aft] = aft_folder

        return gen_folders

    def process_latent_spaces(self, name, model, loader, samples, denoising, suffix):
        opsTr, bottlesTr, e1Tr, e2Tr, d1Tr, d2Tr, yTr = model.collect_reps_data(loader=loader,
                                                                                nSamples=samples,
                                                                                encReps=self.encoderDims,
                                                                                decReps=self.decoderDims,
                                                                                codeLen=self.data_ctrl.intrinsicDim + 1,
                                                                                extrinsicDim=self.data_ctrl.extrinsicDim,
                                                                                denoising=denoising)
        latent_spaces = [self.data_ctrl.Data[1::2], e1Tr, e2Tr, bottlesTr, d1Tr, d2Tr, opsTr]
        layer_names = ["Original", "Enc1", "Enc2", "z", "Dec1", "Dec2", "Rec"]
        global_name = f"{name}_{suffix}"

        aelsprocessor = AELatentSpaceProcessor(latent_spaces=latent_spaces,
                                               layer_names=layer_names,
                                               common_yts=yTr,
                                               global_name=global_name,
                                               current_path=self.current_path,
                                               max_dim=self.max_dim,
                                               metric=self.metric
                                               )

        aelsprocessor.execute()
        aelsprocessor.cleanup()

        del latent_spaces
        del opsTr
        del bottlesTr
        del e1Tr
        del e2Tr
        del d1Tr
        del d2Tr
        del yTr

    def train_model(self, activation=F.relu,
                    name_type=ActivationFunctionTypes.RELU,
                    compute_loss_fct = None,
                    eta = 0.001, epoch=None,
                    samples=2000,
                    denoising=False):
        name = ActivationFunctionTypes.get_name(name_type)
        self.current_path = ActivationFunctionTypes.get_aft_path(self.output_path, name_type)
        if not os.path.isdir(self.current_path):
            os.makedirs(self.current_path)

        model = AE(encoderLayers=self.encoderDims,
                   decoderLayers=self.decoderDims,
                   activation=activation)
        model.to(model.device)
        pretrained_filename = ActivationFunctionTypes.get_aft_pretrained_model_filename(self.output_path, name_type)

        if pretrained_filename is not None:
            model.load_state_dict(torch.load(pretrained_filename))
        else:
            epoch = self.nepoch if epoch is None else epoch
            model = AE(encoderLayers=self.encoderDims,
                      decoderLayers=self.decoderDims,
                       activation=activation)
            model.to(model.device)
            print(model)
            print(model.count_parameters())
            model.my_train(self.data_ctrl.train_loader,
                        n_epochs=epoch,
                        file=f"{self.current_path}/{name}.pt",
                        compute_loss_fct=compute_loss_fct, eta=eta)

        if not AELatentSpaceProcessor.were_pds_computed(self.current_path):
            self.process_latent_spaces(name=name, model=model, loader=self.data_ctrl.train_loader,
                                       samples=samples, denoising=denoising, suffix="train")
            self.process_latent_spaces(name=name, model=model, loader=self.data_ctrl.test_loader,
                                       samples=samples, denoising=denoising, suffix="test")

        del model

    def train_relu(self):
        self.train_model(activation=F.relu, name_type=ActivationFunctionTypes.RELU)

    def train_leaky_relu(self):
        self.train_model(activation=F.leaky_relu, name_type=ActivationFunctionTypes.LRELU)

    def train_tanh(self):
        self.train_model(activation=torch.tanh, name_type=ActivationFunctionTypes.TANH)



class AELatentSpaceProcessor:
    def __init__(self, latent_spaces, layer_names, common_yts,
                 global_name, current_path, max_dim=3, metric="manhattan"):
        self.global_name = global_name
        self.layer_names = layer_names
        self.latent_spaces = latent_spaces
        self.metric = metric
        self.max_dim = max_dim
        self.common_yts = common_yts
        self.current_path = current_path
        if not os.path.isdir(self.current_path):
            os.makedirs(self.current_path)

    def cleanup(self):
        del self.latent_spaces
        del self.layer_names

    def visualization_of_at_latent_codes(self, yTr, bottles, name_fig, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        alphas = [1, 0.3]

        for ii in range(len(np.unique(yTr))):
            idx = (np.where(yTr == ii))[0]
            ax.scatter(bottles[idx, 0], bottles[idx, 1], bottles[idx, 2], alpha=alphas[ii])
            ax.set_axis_off()

        if name_fig is not None:
            plt.savefig(name_fig)

    @staticmethod
    def were_pds_computed(pd_parent_path):
        pdpath = f"{pd_parent_path}/topological_info/"

        all_files = DirectoryHelper.get_all_filenames(root_path=pdpath, file_pattern=".diag")

        return len(all_files) == 14 # 7 layers train + 7 layer test


    def execute(self):
        pd_collection = self.compute_diagrams()
        self.visualize_them_all(pd_collection=pd_collection, legend=True)

        del pd_collection

    def compute_diagrams(self):
        all_diagrams = []

        for id, data in enumerate(self.latent_spaces):
            filename = f"{self.global_name}_persistence_{id}"
            distance_matrix = pairwise_distances(data, n_jobs=-1, metric=self.metric)
            max_value = np.quantile(distance_matrix, 0.75)  # we use the 0.75 quantile as max value

            tda_helper = TDAHelper(input_data_filename=filename,
                                   input_data=distance_matrix,
                                   max_dim=self.max_dim, enable_collapse=self.max_dim,
                                   max_value=max_value, draw_diagrams=False,
                                   output_folder=f"{self.current_path}/topological_info/")

            diagrams = tda_helper.compute_sparse_rips_persistence()

            all_diagrams.append(diagrams)

        return all_diagrams

    def visualize_them_all(self, pd_collection, legend=False):

        self.plot_diagrams(diagrams=pd_collection,
                           fig_name=f"{self.current_path}/pd_layers_{self.global_name}.png",
                           legend=legend)

        self.visualize_latent_space(fig_name=f"{self.current_path}/latent_spaces_{self.global_name}.png")

    def plot_diagrams(self, diagrams, fig_name=None, legend=False):
        n_plots = len(diagrams)
        plt.figure(figsize=(30, 20))
        plot_id = int("1{}1".format(n_plots))
        for i in range(7):
            plt.subplot(plot_id)
            plt.title(self.layer_names[i])
            persim.plot_diagrams(diagrams[i], show=False, legend=legend)
            plot_id += 1
        if fig_name is not None:
            plt.savefig(fig_name)

    def visualize_latent_space(self, fig_name=None):
        n_plots = len(self.latent_spaces)
        plt.figure(figsize=(30, 20))
        plot_id = int("1{}1".format(n_plots))
        for i in range(7):
            ax = plt.subplot(plot_id, projection='3d')
            plt.title(self.layer_names[i])
            self.visualization_of_at_latent_codes(self.common_yts, self.latent_spaces[i], None, ax)
            plot_id += 1
        if fig_name is not None:
            plt.savefig(fig_name)

if __name__ == '__main__':
    output_path = "autoencoder"

    AETrainerController().execute()

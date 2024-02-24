import time

import cv2
import cv2 as cv
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm

from utilities.directoy_helper import DirectoryHelper
from utilities.distance_matrix_helper import DistanceMatrixHelper
from utilities.tda_helper import TDAHelper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImageDataPreprocessor:
    GAUSSIAN, HOG, IGG, NONE = range(4)
    @staticmethod
    def preprocess_gaussianblur(img,
                    bw=True, gray=False, threshold = 0.40,
                    reshape=True, image_target=(128, 128),
                    blur_flag=True, kernel_size=(3, 3),
                    standardize=True):
        """preprocess a single image"""
        if bw:
            img[img < 255 * threshold] = 0
        if gray:
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255
            else:
                img = img / 255
        if reshape:
            img = cv.resize(img, image_target, interpolation=cv.INTER_CUBIC)
        if blur_flag:
            img = cv.GaussianBlur(img, kernel_size, 0)
        if standardize:
            img = (img - np.mean(img)) / np.std(img)
            img = np.array(((img - img.min()) / (img.max() - img.min())) * 255.0, dtype=np.uint8)
            return img

        else:
            return img

    @staticmethod
    def preprocess_none(img,
                                gray=False,
                                reshape=False, image_target=(128, 128),
                                blur_flag=False, kernel_size=(5, 5),
                                standardize=True):
        """preprocess a single image"""
        if gray:
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255
            else:
                img = img / 255
        if reshape:
            img = cv.resize(img, image_target, interpolation=cv.INTER_CUBIC)
        if blur_flag:
            img = cv.GaussianBlur(img, kernel_size, 0)
        if standardize:
            stdv = np.std(img)
            if stdv > 0:
                img = (img - np.mean(img)) / np.std(img)
            else:
                img = (img - np.mean(img))
            img = np.array(((img - img.min()) / (0.00001+img.max() - img.min())) * 255.0, dtype=np.uint8)
            return img
        else:
            return img

    @staticmethod
    def preprocess_hog(img, bw = True, threshold=.40,
                       gray=False,
                       reshape=True, image_target=(128, 128),
                       hog_flag=True, standardize=True):
        from skimage.feature import hog
        """preprocess a single image"""
        if bw:
            img[img < 255 * threshold] = 0

        if gray:
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255
            else:
                img = img / 255
        if reshape:
            img = cv.resize(img, image_target, interpolation=cv.INTER_CUBIC)
        if hog_flag:
            img = hog(img, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2))

        if standardize:
            img = (img - np.mean(img)) / np.std(img)
            img = ((img - img.min()) / (img.max() - img.min())) * 255

            img = img.astype('uint8')
            return img
        else:
            return img

    @staticmethod
    def preprocess_igg(img,bw = True, threshold=.40,
                       gray=False,
                       reshape=True, image_target=(128, 128),
                       igg_flag=True,
                       standardize=True):
        from skimage.segmentation import inverse_gaussian_gradient
        """preprocess a single image"""
        if bw:
            img[img < 255 * threshold] = 0
        if gray:
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255
            else:
                img = img / 255
        if reshape:
            img = cv.resize(img, image_target, interpolation=cv.INTER_CUBIC)
        if igg_flag:
            img = inverse_gaussian_gradient(img)
        if standardize:
            img = (img - np.mean(img)) / np.std(img)
            img = ((img - img.min()) / (img.max() - img.min())) * 255
            img = img.astype('uint8')
            return img
        else:
            return img
    @staticmethod
    def typename(func_type):
        if func_type == ImageDataPreprocessor.IGG:
            return "IGG"
        elif func_type == ImageDataPreprocessor.NONE:
            return "ORIGINAL"
        elif func_type == ImageDataPreprocessor.HOG:
            return "HOG"

        return "GAUSSIAN_BLUR"

    @staticmethod
    def preprocess(xset, func_type=None):
        """Preprocesses the train and test image sets by applying the function
        func[params_dict] to each image. If func is None the function
        self.preprocess_ is used, if params_dict is None,
        :const:`geneo.constants.PREPROC_PARAMS` is used.
        """
        if func_type == ImageDataPreprocessor.GAUSSIAN:
            func = ImageDataPreprocessor.preprocess_gaussianblur
        elif func_type == ImageDataPreprocessor.NONE:
            func = ImageDataPreprocessor.preprocess_none
        elif func_type == ImageDataPreprocessor.HOG:
            func = ImageDataPreprocessor.preprocess_hog

        else:
            func = ImageDataPreprocessor.preprocess_igg

        prep_set = []

        for i, img in enumerate(tqdm(xset, desc="preprocessing train")):
            if type(img) == torch.Tensor:
                img = img.numpy()
            prep_set.append(func(img))

        return np.asarray(prep_set)

    @staticmethod
    def multipleExamplePlots(X, layer, plot_num=10, col=3, name=None, path=None):

        rrow = plot_num % col
        row = plot_num // col
        if rrow != 0:
            row += 1
        name = time.strftime(f"{path}/%y.%m.%d__%H.%M.%S_features_{name}.png")

        plt.figure(figsize=(20, 20))
        for i in range(plot_num):
            # print(layer[i])
            plt.subplot(row, col, i + 1)
            # plt.title(layer[i])
            img = X[i]
            shape_len = len(img.shape)
            if shape_len < 2:
                plt.plot(range(len(img)), img)
            elif shape_len == 2:
                plt.imshow(img)
            else:
                plt.imshow(img[0])
            #plt.colorbar()
        plt.savefig(name)
        plt.show()
        plt.close()


class PytorchDataset(Dataset):
    def __init__(self, dataname, transform_type=None):
        """
        This will download the dataset for on the first use
        """
        module_path = DirectoryHelper.get_module_path()
        data_path = f"{module_path}/datasets/{dataname}"
        transform_type = ImageDataPreprocessor.GAUSSIAN if transform_type is None else transform_type
        self.tname = ImageDataPreprocessor.typename(transform_type)
        train_name = f"{data_path}/{dataname}_train_{self.tname}.npy"
        test_name = f"{data_path}/{dataname}_test_{self.tname}.npy"

        self.xtrain, self.ytrain = self.preprocess_data(data_path, transform_type, train_name, istrain=True)
        self.xtest, self.ytest = self.preprocess_data(data_path, transform_type, test_name, istrain=False)

    def preprocess_data(self, data_path, transform_type, dataset_name, istrain=True):
        if not os.path.isfile(f"{dataset_name}.pickle"):
            if istrain:
                dataset = self.load_data_train(data_path)
            else:
                dataset = self.load_data_test(data_path)

            X, Y = dataset.data, dataset.targets
            if transform_type is not None:
                X = ImageDataPreprocessor.preprocess(X, func_type=transform_type)

            DistanceMatrixHelper.save_distance_matrix((X, Y),
                                                      dataset_name)
            if istrain:
                self.labels = dataset.classes
            del dataset
        else:
            (X, Y) = DistanceMatrixHelper.load_distance_matrix(
                dataset_name)

        return X, Y

    def load_data_train(self, data_path):
        pass
    def load_data_train(self, data_path):
        pass
        
        
class FashionMNist(PytorchDataset):
    def __init__(self, transform_type=None):
        super(FashionMNist, self).__init__(dataname="fashion",
                                           transform_type=transform_type)

    def load_data_train(self, data_path):
        """
        This will download the dataset for on the first use
        """

        train_set = torchvision.datasets.FashionMNIST(data_path,
                                                           download=True,
                                                           transform=transforms.Compose([transforms.ToTensor()]))

        return train_set
    def load_data_train(self, data_path):
        """
        This will download the dataset for on the first use
        """

        test_set = torchvision.datasets.FashionMNIST(data_path,
                                                          download=True,
                                                          train=False,
                                                          transform=transforms.Compose([transforms.ToTensor()]))

        return test_set

class Cifar10(PytorchDataset):
    def __init__(self, transform_type = None):
        """
        This will download the dataset for on the first use
        """
        super(Cifar10, self).__init__(dataname="cifar10", transform_type=transform_type)

    def load_data_train(self, data_path):
        """
        This will download the dataset for on the first use
        """

        train_set = torchvision.datasets.CIFAR10(data_path,
                                                           download=True,
                                                           transform=transforms.Compose([transforms.ToTensor()]))

        return train_set

    def load_data_test(self, data_path):
        """
        This will download the dataset for on the first use
        """

        test_set = torchvision.datasets.CIFAR10(data_path,
                                                download=True,
                                                train=False,
                                                transform=transforms.Compose([transforms.ToTensor()]))

        return test_set

if __name__ == '__main__':
    import traceback
    import os

    global_path = os.environ["PWD"]
    print(global_path)
    CIFAR, FASHION = range(2)
    dataname = {CIFAR: "CIFAR10", FASHION:"FASHION"}

    def test_torch_data(datatype):
        try:
            overall_path = f"results/image_transform/{dataname[datatype]}"
            if not os.path.isdir(overall_path):
                os.makedirs(overall_path)

            if datatype == CIFAR:
                dataOrigin = Cifar10(transform_type = ImageDataPreprocessor.NONE)
                dataGaus = Cifar10(transform_type = ImageDataPreprocessor.GAUSSIAN)
                dataHog = Cifar10(transform_type = ImageDataPreprocessor.HOG)
                dataIgg = Cifar10(transform_type = ImageDataPreprocessor.IGG)
            else:
                dataOrigin = FashionMNist(transform_type=ImageDataPreprocessor.NONE)
                dataGaus = FashionMNist(transform_type=ImageDataPreprocessor.GAUSSIAN)
                dataHog = FashionMNist(transform_type=ImageDataPreprocessor.HOG)
                dataIgg = FashionMNist(transform_type=ImageDataPreprocessor.IGG)

            imgs_ori = []
            imgs_gaus = []
            imgs_hog = []
            imgs_igg = []
            labels = np.unique(dataGaus.ytrain)
            first_in_class = {l: 0 for l in labels}
            filled_class = {l: False for l in labels}
            layer = []
            samples_number = 100
            for id,i in enumerate(dataGaus.ytrain[:samples_number]):
                i = int(i)
                if not filled_class[i]:
                    filled_class[i] = True
                    first_in_class[i] += id
                    imgs_ori.append(dataOrigin.xtrain[i])
                    imgs_gaus.append(dataGaus.xtrain[i])
                    imgs_hog.append(dataHog.xtrain[i])
                    imgs_igg.append(dataIgg.xtrain[i])
            X = []
            X.extend(imgs_ori)
            X.extend(imgs_gaus)
            X.extend(imgs_hog)
            X.extend(imgs_igg)
            # X, layer, plot_num = 10, col = 3, name = None, path = None
            layer = ["Original",  "GaussianBlur", "HOG", "Inverse Gaussian Gradient"]
            name = "features_comparison"
            ImageDataPreprocessor.multipleExamplePlots(X, list(labels)*(len(X)//10), plot_num=len(X),
                                                       col=10, name=name, path=overall_path)
            diagrams = []
            cifarobj = [dataOrigin, dataGaus, dataHog, dataIgg]
            for l in first_in_class:
                id = first_in_class[l]
                for j in range(len(cifarobj)):
                    img = cifarobj[j].xtrain[id]
                    filename = f"pd_train_{id}"
                    tda_helper = TDAHelper(distance_matrix_filename=filename,
                                           custom_distance_matrix=img,
                                           max_dim=2, draw_diagrams=True,
                                           output_folder=f"{overall_path}/{layer[j]}/topological_info/")
                    diag = tda_helper.compute_diagrams_by_type(TDAHelper.CUBICAL)
                    diagrams.append(diag)


        except Exception as e:
            print(e)
            traceback.print_exc()


    test_torch_data(CIFAR)
    # test_torch_data(FASHION)
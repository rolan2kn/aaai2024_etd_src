import os.path

from utilities.data_processors_helper import DataProcessorFactory
from utilities.tda_helper import TDAHelper
from utilities.ae_handler import AETrainerController
from experiments.ae_experiment import AEExperiment
from experiments.outex_experiment import OutexExperiment
from experiments.shrec07_experiment import Shrec07Experiment
from experiments.fashion_experiment import FashionExperiment
'''
An Application represents a domain controller that makes use 
of data processors, helpers and experiments in 
order to perform an analysis. We manage two applications:

SupervisedLearningApp and AutoencoderWeightTopologyApp.

  
'''

class Application:
    def __init__(self, overall_path, metric=None):
        self.overall_path = overall_path
        self.metric = metric if metric is not None else "euclidean"
        self.app_name = str(self.__class__.__name__)
        self.output_path = f"{self.overall_path}/{self.app_name}/"

    def build_persistence_diagrams(self):
        pass

    def run_experiments(self):
        pass

    def execute(self):
        self.build_persistence_diagrams()
        self.run_experiments()

    def get_app_name(self):
        return

class SupervisedLearningApp(Application):
    def __init__(self, overall_path, metric = "euclidean"):
        super(SupervisedLearningApp, self).__init__(overall_path, metric)

        self.data_processors = {}
        for data_type in [DataProcessorFactory.OUTEX,
                          # DataProcessorFactory.FASHION,
                          # DataProcessorFactory.FASHION_GAUSS,
                          DataProcessorFactory.FASHION_HOG,
                          # DataProcessorFactory.FASHION_IGG,
                          DataProcessorFactory.SHREC07]:
            dp = DataProcessorFactory.get_data_processor(overall_path=self.output_path, data_type=data_type)
            self.data_processors.update({data_type: dp})

    def build_persistence_diagrams(self):
        max_dim = 3
        for data_type in self.data_processors:
            dp = self.data_processors[data_type]
            TDAHelper.compute_all_diagrams(dp, max_dim)

    def run_experiments(self):
        '''
        '''
        bag_of_experiments = {DataProcessorFactory.OUTEX: OutexExperiment,
                              DataProcessorFactory.FASHION: FashionExperiment,
                              DataProcessorFactory.SHREC07: Shrec07Experiment}

        for data_type in self.data_processors:
            # if there is no match between experiment
            if data_type not in bag_of_experiments:
                continue

            dp = self.data_processors[data_type]
            experiment_class = bag_of_experiments[data_type]
            dp.execute() # if it was already executed this method do nothing
            dp_settings = DataProcessorFactory.get_data_processor_settings(data_type)

            exp_obj = experiment_class(data_processor=dp,
                                       p=dp_settings["p"],
                                       no_train_samples=dp_settings["no_train_samples"],
                                       no_test_samples=dp_settings["no_test_samples"])
            exp_obj.execute() # this perform the knn classifier


class AutoencoderWeightTopologyApp(Application):
    def __init__(self, overall_path, metric='manhattan'):
        super(AutoencoderWeightTopologyApp, self).__init__(overall_path, metric)

        self.ae_controller = AETrainerController(output_path=self.output_path, metric=self.metric)

    def build_persistence_diagrams(self):
        """
        For each model type RELU, LRELU, TANH and DAE
        it generates the autoencoder input data,
        trains the model, and process the latent space of each layer
        generating images and persistence diagrams.
        This function acts as a data generator and processor.

        If all data was already computed the computational load is minimal.
        """
        self.ae_controller.execute()

    def run_experiments(self):
        policies = [AEExperiment.VERSUS_FIRST, AEExperiment.VERSUS_PREVIOUS]
        for current_policy in policies:
            aeexp_obj = AEExperiment(self.ae_controller, p=2, policy=current_policy)
            aeexp_obj.execute()


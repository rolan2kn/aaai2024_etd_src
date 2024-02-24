import time
import os
import numpy as np
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import  classification_report
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV, KFold
from sklearn import neighbors

from utilities.distance_matrix_helper import DistanceMatrixHelper
from experiments.sl_experiments import SLExperiment
from utilities.data_processors_helper import DatasetProcessor
from utilities.directoy_helper import DirectoryHelper
from utilities.knn_classifier_helper import KNearestNeighborHelper
from utilities.tda_helper import TDAHelper


class Cifar10Experiment(SLExperiment):
    def __init__(self, data_processor, p = 2,
                 no_train_samples = None,
                 no_test_samples = None,
                 save_diags = False):
        super(Cifar10Experiment, self).__init__(data_processor, p, no_train_samples,
                                                no_test_samples, save_diags)
        self.save_all = False

    def execute(self):
        knn = KNearestNeighborHelper(self, train_flag=False)
        knn.execute()

if __name__ == '__main__':
    import traceback
    import os

    global_path = os.environ["PWD"]
    print(global_path)

    try:
        from utilities.data_processors_helper import DataProcessorFactory

        overall_path = "results/SupervisedLearningApp/"

        cifar10_processor = DataProcessorFactory.get_data_processor(overall_path=overall_path,
                                                                  data_type=DataProcessorFactory.CIFAR10_GAUSS)
        cifar10_processor.execute()
        # c10e = Cifar10Experiment(data_processor=cifar10_processor, p=2, no_train_samples=3, no_test_samples=1)
        c10e = Cifar10Experiment(data_processor=cifar10_processor, p=1, no_train_samples=100, no_test_samples=10)
        # c10e.get_train_test_data()
        # c10e.compute_distance_matrices([DistanceMatrixHelper.ETDA_H1])
        c10e.execute()
    except Exception as e:
        print(e)
        traceback.print_exc()


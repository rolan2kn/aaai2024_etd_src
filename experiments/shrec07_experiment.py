import numpy as np
from experiments.sl_experiments import SLExperiment
from utilities.directoy_helper import DirectoryHelper
from utilities.knn_classifier_helper import KNearestNeighborHelper
from utilities.tda_helper import TDAHelper


class Shrec07Experiment(SLExperiment):
    def __init__(self, data_processor, p = 2,
                 no_train_samples = None,
                 no_test_samples = None,
                 save_diags = False):
        super(Shrec07Experiment, self).__init__(data_processor, p, no_train_samples,
                                                no_test_samples, save_diags)
        self.save_all = False

    def init_global_files(self):
        train_indexes, test_indexes = self.data_processor.generate_subsampling_indexes(self.no_train_samples,
                                                                                       self.no_test_samples)
        if SLExperiment.train_pdiag_files is None:
            pdiag_files_train = DirectoryHelper.get_all_filenames(self.topological_path,
                                                                  file_pattern="pd_train",
                                                                  ignore_pattern=".png")
            pdiag_files_train = DirectoryHelper.sort_filenames_by_suffix(pdiag_files_train, sep="_")

            if len(train_indexes):
                pdiag_files_train = np.array(pdiag_files_train)
                pdiag_files_train = pdiag_files_train[train_indexes]

            SLExperiment.train_pdiag_files = pdiag_files_train

        if SLExperiment.test_pdiag_files is None:
            # we use the train set to infer the test set using the test_indexes
            pdiag_files_test = DirectoryHelper.get_all_filenames(self.topological_path,
                                                                 file_pattern="pd_train",
                                                                 ignore_pattern=".png")
            pdiag_files_test = DirectoryHelper.sort_filenames_by_suffix(pdiag_files_test, sep="_")

            if len(test_indexes):
                pdiag_files_test = np.array(pdiag_files_test)
                pdiag_files_test = pdiag_files_test[test_indexes]

            SLExperiment.test_pdiag_files = pdiag_files_test

        if SLExperiment.train_pdiags is None:
            # concatenates the train and test set to process them together
            output_dir = None
            if self.save_diags:
                output_dir = f"{self.topological_path}/train_pdiags"

            SLExperiment.train_pdiags = TDAHelper.get_all_pdiagrams(SLExperiment.train_pdiag_files, output_dir)

        if SLExperiment.test_pdiags is None and self.data_processor.has_a_test_set():
            # concatenates the train and test set to process them together
            output_dir = None
            if self.save_diags:
                output_dir = f"{self.topological_path}/test_pdiags"

            SLExperiment.test_pdiags = TDAHelper.get_all_pdiagrams(SLExperiment.test_pdiag_files, output_dir)
        return

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

        shrec_processor = DataProcessorFactory.get_data_processor(overall_path=overall_path,
                                                                  data_type=DataProcessorFactory.SHREC07)
        shrec_processor.execute()
        shrec = Shrec07Experiment(data_processor=shrec_processor,
                                  p=2,
                                  no_train_samples=15,
                                  no_test_samples=10)
                                  # no_test_samples=50) # we will take 5 samples as test set
        # shrec.compute_distance_matrices()
        shrec.execute()
    except Exception as e:
        print(e)
        traceback.print_exc()


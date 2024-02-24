import time

import numpy as np

from experiments.base_experiments import BaseExperiment
from utilities.directoy_helper import DirectoryHelper
from utilities.distance_matrix_helper import DistanceMatrixHelper
from utilities.extended_topology_distance import ExtendedTopologyDistanceHelper
from utilities.tda_helper import TDAHelper


class SLExperiment(BaseExperiment):
    train_pdiag_files = None
    test_pdiag_files = None
    train_pdiags = None
    test_pdiags = None

    def __init__(self, data_processor, p=2,
                 no_train_samples=None,
                 no_test_samples=None,
                 save_diags=False):

        self.data_processor = data_processor
        result_folder = self.data_processor.get_result_folder()
        super(SLExperiment, self).__init__(result_folder=result_folder,
                                           p=2,
                                           save_diags=save_diags)

        self.no_train_samples = no_train_samples
        self.no_test_samples = no_test_samples
        self.topological_path = f"{self.td_path}/{self.topological_folder}/"
        self.validate_topological_path()

        self.td_path = f"{self.td_path}/{self.experiment_name}/{self.distance_folder}"

        self.result_folder = result_folder
        self.save_diags = save_diags
        self.p = p

        self.init_global_files()
        self.distance_collection = {}

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
            if self.no_train_samples:
                self.no_train_samples = len(pdiag_files_train)

            SLExperiment.train_pdiag_files = pdiag_files_train

        if SLExperiment.test_pdiag_files is None:
            pdiag_files_test = DirectoryHelper.get_all_filenames(self.topological_path,
                                                                 file_pattern="pd_test",
                                                                 ignore_pattern=".png")
            pdiag_files_test = DirectoryHelper.sort_filenames_by_suffix(pdiag_files_test, sep="_")

            if len(test_indexes):
                pdiag_files_test = np.array(pdiag_files_test)
                pdiag_files_test = pdiag_files_test[test_indexes]
            if self.no_test_samples is None:
                self.no_test_samples = len(pdiag_files_test)

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

    def retrieve_train_test_pdiags(self, train_indexes,
                                   test_indexes):

        X_train = [SLExperiment.train_pdiags[i] for i in train_indexes]
        X_test = [SLExperiment.train_pdiags[i] for i in test_indexes]

        return X_train, X_test

    def get_train_test_data(self):
        x_train = SLExperiment.train_pdiags
        x_test = SLExperiment.test_pdiags

        y_train = self.data_processor.get_chosen_train_labels()
        y_test = self.data_processor.get_chosen_test_labels()

        return x_train, x_test, y_train, y_test

    def get_train_test_label(self):
        """
        At this point the data processor was already executed
        and desired indexes already computed. Therefore, it is safe
        to ask for desired labels
        """
        y_train = self.data_processor.get_chosen_train_labels()
        y_test = self.data_processor.get_chosen_test_labels()

        return y_train, y_test

    def get_total_size(self):
        train_size = self.no_train_samples if self.no_train_samples is not None else len(SLExperiment.train_pdiags)
        test_size = self.no_test_samples if self.no_test_samples is not None else len(SLExperiment.test_pdiags)

        total_size = len(self.data_processor.get_classes()) * train_size * test_size
        return total_size
    def get_train_test_pdiags_from_indexes(self):

        x_train = SLExperiment.train_pdiags
        x_test = SLExperiment.test_pdiags
        y_train = self.data_processor.get_chosen_train_labels()
        y_test = self.data_processor.get_chosen_test_labels()

        return x_train, x_test, y_train, y_test

    def execute(self, desired_distance):
        pass

    def compute_distance_matrices(self, dtypes=None, train=None, test=None, fold_no=0):
        angle_set = ExtendedTopologyDistanceHelper.compute_angle_dict()
        slice_set = ExtendedTopologyDistanceHelper.compute_swd_slices()

        distance_folder = self.td_path
        file_name = time.strftime(
            "{0}%y.%m.%d__%H.%M.%S_time_result_file.txt".format(distance_folder))
        result_file = open(file_name, "w")
        distance_types = DistanceMatrixHelper.get_all_distance_types() if dtypes is None else dtypes

        if train is None:
            train = SLExperiment.train_pdiags
        if test is None:
            test = SLExperiment.test_pdiags

        test_size = len(test)
        train_size = len(train)

        if test_size == 0 or train_size == 0:
            return

        if not hasattr(self, "distance_collection"):
            self.distance_collection = {}

        for dtype in distance_types:
            try:
                if not DistanceMatrixHelper.is_swd_type(dtype):
                    value_set = angle_set
                else:
                    value_set = slice_set
                '''
                 compute_distance_matrix_test_vs_train(distance_type: int, test_pds: list, 
                 train_pds: list,
                                              p: int, angle_set: dict,
                                              distance_folder: str, size: int=None
                '''
                dist, avg_time = DistanceMatrixHelper.compute_distance_matrix_test_vs_train(
                    distance_type=dtype,
                    test_pds=test,
                    train_pds=train,
                    p=self.p,
                    angle_set=value_set,
                    distance_folder=distance_folder, fold_no=fold_no
                )
                self.distance_collection[dtype] = dist
                if avg_time is not None:
                    if dtype not in (DistanceMatrixHelper.ETDA_H1,
                                     DistanceMatrixHelper.ETDA_METRIC_H1,
                                     DistanceMatrixHelper.SWD_H1):
                        result_file.write(f"{DistanceMatrixHelper.distance_names[dtype]}: {avg_time} ms\n")
                    else:
                        for vkey in value_set:
                            result_file.write(
                                f"{DistanceMatrixHelper.distance_names[dtype]}_{vkey}: {avg_time[vkey]} ms\n")
                del dist
            except Exception as e:
                print(e)
                self.distance_collection[dtype] = None
        result_file.close()

        return

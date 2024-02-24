# from pathos.multiprocessing import ProcessPool
import multiprocessing
# dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
# multiprocessing.reduction.ForkingPickler = dill.Pickler
# multiprocessing.reduction.dump = dill.dump
# multiprocessing.queues._ForkingPickler = dill.Pickler
import os
import os.path
import time

import numpy as np


class TTDistanceMatrixExecutor:
    '''
    This class compute distance matrices using the python multiprocessing package.

    Essentially, it makes available the set of items to compare and assistance data, to avoid
     passing them to each task. We also delve on the upper triangular structure of the expected
     distance matrices, considered as a list, but we resolve indexes of desired i,j positions
     then we access to shared objects and compute the distances.

     All distance matrices are assumed to be in one homology group typically H1

     TODO: migrate this to pytorch multiprocessing and distributed packages
    '''
    TRAIN_PDS = None
    TEST_PDS = None
    P = 2
    ROWS = None
    COLS = None
    DTYPE = None
    ANGLE_SET = None
    MAX_DIM = None

    def __init__(self, train_pds: list, test_pds: list,
                 dtype: int, max_dim: int = 3,
                 p: int = 2, angle_set: dict = {}, whole_distance=False, fixed_rows=[]):

        TTDistanceMatrixExecutor.TRAIN_PDS = train_pds
        TTDistanceMatrixExecutor.TEST_PDS = test_pds
        TTDistanceMatrixExecutor.ROWS = len(test_pds)
        TTDistanceMatrixExecutor.COLS = len(train_pds)

        TTDistanceMatrixExecutor.FIXED_ROWS = fixed_rows
        TTDistanceMatrixExecutor.DTYPE = dtype
        TTDistanceMatrixExecutor.P = p
        TTDistanceMatrixExecutor.MAX_DIM = self.max_dim = max_dim
        TTDistanceMatrixExecutor.ANGLE_SET = angle_set
        self.whole_distance = whole_distance

    @staticmethod
    def inverse_indexes(idx):
        M = TTDistanceMatrixExecutor.COLS
        i = idx // M
        j = idx % M

        return i, j

    @staticmethod
    def compute_distance(idx):
        from utilities.distance_matrix_helper import DistanceMatrixHelper

        i, j = TTDistanceMatrixExecutor.inverse_indexes(idx)
        dtype = TTDistanceMatrixExecutor.DTYPE

        diag1 = TTDistanceMatrixExecutor.TEST_PDS[i]
        diag2 = TTDistanceMatrixExecutor.TRAIN_PDS[j]

        distance_function = DistanceMatrixHelper.distance_functions[dtype]
        t1 = time.time_ns()
        result = distance_function(diag1, diag2, TTDistanceMatrixExecutor.P)
        item_time = (time.time_ns() - t1) * (10 ** -6)

        return idx, result, item_time

    @staticmethod
    def compute_distance_with_angles(idx):
        from utilities.distance_matrix_helper import DistanceMatrixHelper

        i, j = TTDistanceMatrixExecutor.inverse_indexes(idx)
        dtype = TTDistanceMatrixExecutor.DTYPE

        diag1 = TTDistanceMatrixExecutor.TEST_PDS[i]
        diag2 = TTDistanceMatrixExecutor.TRAIN_PDS[j]

        angle_set = TTDistanceMatrixExecutor.ANGLE_SET

        distance_function = DistanceMatrixHelper.distance_functions[dtype]
        average_time = {}
        dist = {}
        for aname in angle_set:
            average_time[aname] = 0
            t1 = time.time_ns()
            etda = distance_function(diag1, diag2, TTDistanceMatrixExecutor.P, angle_set[aname])
            average_time[aname] += (time.time_ns() - t1) * (10 ** -6)  # time in milliseconds
            dist.update({aname: etda})

        return idx, dist, average_time

    def is_common(self):
        from utilities.distance_matrix_helper import DistanceMatrixHelper

        return DistanceMatrixHelper.is_common_type(TTDistanceMatrixExecutor.DTYPE)

    def process_common_distances(self):
        N = TTDistanceMatrixExecutor.ROWS
        M = TTDistanceMatrixExecutor.COLS
        total_elements = N * M
        avg_time = 0
        dmatrix = np.zeros((N, M), dtype=np.float16)
        no_cores = len(os.sched_getaffinity(0))
        with multiprocessing.Pool(no_cores) as pool:
            for idx, dist, row_avg_time in pool.imap_unordered(TTDistanceMatrixExecutor.compute_distance,
                                                               range(0, total_elements)):
                i, j = TTDistanceMatrixExecutor.inverse_indexes(idx)
                avg_time += row_avg_time
                dmatrix[i][j] = dist

        avg_time /= total_elements
        return dmatrix, avg_time

    def process_angle_distances(self):
        N = TTDistanceMatrixExecutor.ROWS
        M = TTDistanceMatrixExecutor.COLS
        total_elems = N * M
        avg_time = {}
        dmatrix = {}

        for aname in TTDistanceMatrixExecutor.ANGLE_SET:
            avg_time.update({aname: 0})
            dmatrix.update({aname: np.zeros((N, M), dtype=np.float16)})

        no_cores = len(os.sched_getaffinity(0))
        with multiprocessing.Pool(no_cores) as pool:

            for idx, dist, row_avg_time in pool.imap_unordered(TTDistanceMatrixExecutor.compute_distance_with_angles,
                                                               range(0, total_elems)):
                i, j = TTDistanceMatrixExecutor.inverse_indexes(idx)
                for aname in TTDistanceMatrixExecutor.ANGLE_SET:
                    avg_time[aname] += row_avg_time[aname]
                    dmatrix[aname][i][j] = dist[aname]

        for aname in TTDistanceMatrixExecutor.ANGLE_SET:
            avg_time[aname] /= total_elems

        return dmatrix, avg_time

    def process_common_distances_sequential(self):
        from utilities.distance_matrix_helper import DistanceMatrixHelper
        dtype = TTDistanceMatrixExecutor.DTYPE

        N = TTDistanceMatrixExecutor.ROWS
        M = TTDistanceMatrixExecutor.COLS
        total_elements = N * M
        avg_time = 0
        distance_function = DistanceMatrixHelper.distance_functions[dtype]

        dmatrix = np.zeros((TTDistanceMatrixExecutor.ROWS, TTDistanceMatrixExecutor.COLS), dtype=np.float16)

        for i in range(N):
            diagi = TTDistanceMatrixExecutor.TEST_PDS[i]
            for j in range(M):
                diagj = TTDistanceMatrixExecutor.TRAIN_PDS[j]

                t1 = time.time_ns()
                dmatrix[i][j] = distance_function(diagi, diagj, TTDistanceMatrixExecutor.P)
                avg_time += (time.time_ns() - t1) * (10 ** -6)  # time in milliseconds

        avg_time /= total_elements

        return dmatrix, avg_time

    def process_angle_distances_sequential(self):
        from utilities.distance_matrix_helper import DistanceMatrixHelper
        dtype = TTDistanceMatrixExecutor.DTYPE
        angle_set = TTDistanceMatrixExecutor.ANGLE_SET

        N = TTDistanceMatrixExecutor.ROWS
        M = TTDistanceMatrixExecutor.COLS
        total_elements = N * M
        avg_time = {}
        dmatrix = {}
        distance_function = DistanceMatrixHelper.distance_functions[dtype]

        for aname in TTDistanceMatrixExecutor.ANGLE_SET:
            avg_time.update({aname: 0})
            dmatrix.update({aname: np.zeros((N, M), dtype=np.float16)})

        for i in range(N):
            diagi = TTDistanceMatrixExecutor.TEST_PDS[i]
            for j in range(M):
                diagj = TTDistanceMatrixExecutor.TRAIN_PDS[j]
                for aname in TTDistanceMatrixExecutor.ANGLE_SET:
                    t1 = time.time_ns()
                    dmatrix[aname][i][j] = distance_function(diagi, diagj, TTDistanceMatrixExecutor.P, angle_set[aname])
                    avg_time[aname] += (time.time_ns() - t1) * (10 ** -6)  # time in milliseconds

        for aname in TTDistanceMatrixExecutor.ANGLE_SET:
            avg_time[aname] /= total_elements

        return dmatrix, avg_time

    def execute(self):
        if self.is_common():
            return self.process_common_distances()

        return self.process_angle_distances()

    def execute_sequential(self):
        if self.is_common():
            return self.process_common_distances_sequential()

        return self.process_angle_distances_sequential()


if __name__ == '__main__':
    from utilities.directoy_helper import DirectoryHelper
    from utilities.distance_matrix_helper import DistanceMatrixHelper
    from utilities.extended_topology_distance import ExtendedTopologyDistanceHelper
    from utilities.tda_helper import TDAHelper


    def test_distance_computation():
        all_folders = DirectoryHelper.get_all_subfolders(root_path="results/SupervisedLearningApp/",
                                                         dir_pattern="Processor")
        if len(all_folders) == 0:
            print("We cannot exec our test, since you have no any data processor ")
            exit(0)

        overall_path = f"{all_folders[0]}/topological_info"
        train_pd_names = DirectoryHelper.get_all_filenames(root_path=overall_path,
                                                           file_pattern="pd_train")

        train_pd_names = DirectoryHelper.sort_filenames_by_suffix(train_pd_names, sep="_")

        test_pd_names = DirectoryHelper.get_all_filenames(root_path=overall_path,
                                                          file_pattern="pd_test")

        test_pd_names = DirectoryHelper.sort_filenames_by_suffix(test_pd_names, sep="_")
        # try with 20 elements
        train_pd = TDAHelper.get_all_pdiagrams(pd_all_diagnames=train_pd_names[:1000],
                                               output_filename=None)
        test_pd = TDAHelper.get_all_pdiagrams(pd_all_diagnames=test_pd_names[:10],
                                              output_filename=None)

        angle_set = ExtendedTopologyDistanceHelper.compute_angle_dict()
        slice_set = ExtendedTopologyDistanceHelper.compute_swd_slices()
        all_dtypes = DistanceMatrixHelper.get_all_distance_types()
        value_set = {}
        for dtype in all_dtypes:
            if dtype in (DistanceMatrixHelper.ETDA_H1, DistanceMatrixHelper.ETDA_METRIC_H1):
                value_set = angle_set
            elif dtype == DistanceMatrixHelper.SWD_H1:
                value_set = slice_set
            dname = DistanceMatrixHelper.distance_names[dtype]
            print(f"***~~*~~*~~*~~*~~*~~*~~*~~*~~*** Processing distance {dname} ***~~*~~*~~*~~*~~*~~*~~*~~*~~***")
            magic_dm = TTDistanceMatrixExecutor(train_pds=train_pd,
                                                test_pds=test_pd,
                                                dtype=dtype,
                                                max_dim=2,
                                                p=2,
                                                angle_set=value_set,
                                                whole_distance=True)
            t1 = time.time_ns()
            M1, avg_time1 = magic_dm.execute()
            duration1 = (time.time_ns() - t1) * (10 ** -6)


            t1 = time.time_ns()
            M2, avg_time2 = magic_dm.execute_sequential()
            duration2 = (time.time_ns() - t1) * (10 ** -6)
            # print(f"matrix: {M1} avg_time: {avg_time} ms total_time: {duration1} ms")
            # print(f"seq_matrix: {M2} avg_time: {avg_time} ms total_time: {duration} ms")
            print(f"both are equal: {M1 == M2}\n")
            print(f"parallel_matrix M1: avg_time: {avg_time1} ms total_time: {duration1} ms\n")
            print(f"seq_matrix M2: avg_time: {avg_time2} ms total_time: {duration2} ms\n")


    test_distance_computation()

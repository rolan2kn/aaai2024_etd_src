import multiprocessing
import os
import pickle
import time
from multiprocessing import Pool

import gudhi.representations
import ot
import persim
import numpy as np

from utilities.tt_distance_matrix_executor import TTDistanceMatrixExecutor
from utilities.extended_topology_distance import ExtendedTopologyDistanceHelper, FastExtendedTopologyDistanceHelper
from utilities.persistence_diagram_helper import PersistenceDiagramHelper

from gudhi.hera import wasserstein_distance as hera_wd
from gudhi.representations import Entropy

class DistanceMatrixHelper:
    (BASIC_ETD, ETDA, ETDA_METRIC, WD, POT_WD, HERA_WD, SWD, PS, FISHER,
     BASIC_ETD_H1, ETDA_H1, ETDA_METRIC_H1, WD_H1, POT_WD_H1, HERA_WD_H1, SWD_H1, PS_H1, FISHER_H1,
     NP_BASIC_ETD, NP_ETDA, NP_ETDA_METRIC, NP_BASIC_ETD_H1, NP_ETDA_H1, NP_ETDA_METRIC_H1,) = range(24)

    distance_names = {BASIC_ETD: "BASIC_ETD",
                      ETDA: "ETD_",
                      ETDA_METRIC: "ETD_METRIC_",
                      WD: "WD", HERA_WD: "HERA_WD",
                      SWD: "SWD", PS: 'PS', FISHER: 'FISHER',
                      BASIC_ETD_H1: "BASIC_ETD_H1",
                      ETDA_H1: "ETD_H1_",
                      ETDA_METRIC_H1: "ETD_METRIC_H1_",
                      WD_H1: "WD_H1", HERA_WD_H1: "HERA_WD_H1",
                      SWD_H1: "SWD_H1_", PS_H1: 'PS_H1', FISHER_H1: 'FISHER_H1',
                      NP_BASIC_ETD: "NP_BASIC_ETD",
                      NP_ETDA: "NP_ETD_",
                      NP_ETDA_METRIC: "NP_ETD_METRIC",
                      NP_BASIC_ETD_H1: "NP_BASIC_ETD_H1",
                      NP_ETDA_H1: "NP_ETD_H1_",
                      NP_ETDA_METRIC_H1: "NP_ETD_METRIC_H1_"}

    @staticmethod
    def get_all_distance_types():
        return [DistanceMatrixHelper.BASIC_ETD_H1,
                DistanceMatrixHelper.ETDA_H1,
                DistanceMatrixHelper.NP_BASIC_ETD_H1,
                DistanceMatrixHelper.NP_ETDA_H1,
                DistanceMatrixHelper.WD_H1,
                DistanceMatrixHelper.HERA_WD_H1,
                DistanceMatrixHelper.SWD_H1,
                DistanceMatrixHelper.PS_H1,
                DistanceMatrixHelper.FISHER_H1]
    @staticmethod
    def save_data_collection(data_collection, filename, extension = None):
        if data_collection is None:
            raise Exception("We were expected a valid data collection. Please review your argument")
        if len(data_collection) == 0:
            return []

        if extension is None:
            extension = ".pickle"
        filename = f"{filename}{extension}" if filename.find(extension) == -1 else filename

        pickle_dist_out = open(filename, "wb")
        pickle.dump(data_collection, pickle_dist_out)
        pickle_dist_out.close()

        return data_collection

    @staticmethod
    def load_data_collection(filename, extension = None):
        if extension is None:
            extension = ".pickle"
        filename = f"{filename}{extension}" if filename.find(extension) == -1 else filename
        if os.path.isfile(filename):
            # to load it please use:
            pickle_dist_in = open(filename, "rb")
            data_collection = pickle.load(pickle_dist_in)
            pickle_dist_in.close()

            return data_collection

        return None

    @staticmethod
    def save_distance_matrix(distance_matrix, filename):
        return DistanceMatrixHelper.save_data_collection(data_collection=distance_matrix,
                                                         filename=filename)

    @staticmethod
    def load_distance_matrix(filename):
        return DistanceMatrixHelper.load_data_collection(filename=filename)

    @staticmethod
    def resolve_pd_distance_name(dtype: int, size: int, angle_set: dict, max_dim: int, distance_folder: str, prefix: str=""):
        dist_name = f"{distance_folder}/{prefix}{DistanceMatrixHelper.distance_names[dtype]}_{max_dim}_{size}"
        if dtype in (DistanceMatrixHelper.ETDA_H1, DistanceMatrixHelper.ETDA_METRIC_H1, DistanceMatrixHelper.SWD_H1):
            angle_set_name = "_".join(angle_set.keys())
            dist_name = f"{dist_name}_{angle_set_name}"

        return f"{dist_name}.pickle"

    @staticmethod
    def store_value_on_distance_matrix(distance_dict, dtype, d, angle, value):
        if d not in distance_dict:
            distance_dict.update({d: None})

        if dtype in (DistanceMatrixHelper.ETDA, DistanceMatrixHelper.ETDA_METRIC):
            if distance_dict[d] is None:
                distance_dict[d] = {}
            if angle not in distance_dict[d]:
                distance_dict[d].update({angle: None})
            distance_dict[d][angle] = value
        else:
            distance_dict[d] = value

    @staticmethod
    def get_corresponding_distance(dtype):
        return DistanceMatrixHelper.distance_functions[dtype]

    @staticmethod
    def compute_pd_distance_matrix_with_diagrams(distance_type: int, all_pds: list, p: int, angle_set: dict,
                                   distance_folder: str, size: int=None):
        '''
        This method compute desired distances on the persistence diagram space.

        This is the bottleneck! So we use the TTDistanceMatrixExecutor
        '''
        if size is None:
            size = len(all_pds)
        if size == 0:
            return {}, None

        # load the first diagram to determine the dimension, this assumes that all diagrams
        # have the same number of homology groups
        max_sizes = PersistenceDiagramHelper.get_maximum_sizes(all_pds)
        max_dim = len(max_sizes)

        '''
        Verify if desired distances were already computed.
        To achieve that, we resolve the distance names
        considering all parameters of the desired distance matrices,
        since params are the same for the same matrices if computed 
        it will be present on the distance_folder.
        '''
        distance_name = DistanceMatrixHelper.resolve_pd_distance_name(distance_type,
                                                                      size,
                                                                      angle_set,
                                                                      max_dim,
                                                                      distance_folder)
        if os.path.isfile(distance_name):
            return DistanceMatrixHelper.load_distance_matrix(distance_name), None
            #dm_storer = DistanceMatrixHDF5Storer.load_hdf5_from_file(distance_matrix_filename=distance_name)

            #return dm_storer.get_distance_matrix(), None

        # resulting dictionaries
        # verify existence of distance folder and create it if needed
        dm_path = f"{distance_folder}/"
        if not os.path.isdir(dm_path):
            os.makedirs(dm_path)

        fast_dm_executor = TTDistanceMatrixExecutor(all_pds = all_pds,
                                                      dtype=distance_type,
                                                      max_dim=max_dim,
                                                      p=p,
                                                      angle_set=angle_set,
                                                      whole_distance=True)

        t1 = time.time_ns()
        distance_matrix, avg_time = fast_dm_executor.execute()
        t2 = time.time_ns() - t1

        print(f"parallel time: {t2}, par_avg_time: {avg_time} \n")

        DistanceMatrixHelper.save_distance_matrix(distance_matrix, distance_name)
        # dm_storer = DistanceMatrixHDF5Storer(storage_path=distance_folder,
        #                                      distance_matrix_name=distance_name,
        #                                      distance_matrix=distance_matrix)
        return distance_matrix, avg_time#dm_storer.get_distance_matrix(), avg_time

    @staticmethod
    def compute_distance_matrix_test_vs_train(distance_type: int,
                                              test_pds: list,
                                              train_pds: list,
                                              p: int,
                                              angle_set: dict,
                                              distance_folder: str, fold_no = 0):
        '''
        This method compute desired distances on the persistence diagram space.

        This is the bottleneck! So we use the TTDistanceMatrixExecutor
        '''

        # load the first diagram to determine the dimension, this assumes that all diagrams
        # have the same number of homology groups
        train_max_sizes = PersistenceDiagramHelper.get_maximum_sizes(train_pds)
        test_max_sizes = PersistenceDiagramHelper.get_maximum_sizes(test_pds)
        max_dim = max(len(train_max_sizes), len(test_max_sizes))

        '''
        Verify if desired distances were already computed.
        To achieve that, we resolve the distance names
        considering all parameters of the desired distance matrices,
        since params are the same for the same matrices if computed 
        it will be present on the distance_folder.
        '''
        distance_name = DistanceMatrixHelper.resolve_pd_distance_name(distance_type,
                                                                      len(test_pds)*len(train_pds),
                                                                      angle_set,
                                                                      max_dim,
                                                                      distance_folder,
                                                                      prefix=f"tyt_{fold_no}_")
        if os.path.isfile(distance_name):
            return DistanceMatrixHelper.load_distance_matrix(distance_name), None

        # resulting dictionaries
        # verify existence of distance folder and create it if needed
        dm_path = f"{distance_folder}/"
        if not os.path.isdir(dm_path):
            os.makedirs(dm_path)

        dm_executor = TTDistanceMatrixExecutor(train_pds = train_pds, test_pds = test_pds,
                                             dtype=distance_type,
                                             max_dim=max_dim,
                                             p=p,
                                             angle_set=angle_set,
                                             whole_distance=True)

        t1 = time.time_ns()
        distance_matrix, avg_time = dm_executor.execute_sequential()
        # distance_matrix, avg_time = dm_executor.execute()
        t2 = time.time_ns() - t1

        print(f"parallel time: {t2}, par_avg_time: {avg_time} \n")

        DistanceMatrixHelper.save_distance_matrix(distance_matrix, distance_name)
        # dm_storer = DistanceMatrixHDF5Storer(storage_path=distance_folder,
        #                                      distance_matrix_name=distance_name,
        #                                      distance_matrix=distance_matrix)
        return distance_matrix, avg_time#dm_storer.get_distance_matrix(), avg_time

    @staticmethod
    def compute_pd_swd_distance(diag1, diag2, p=2, slice=50):
        '''
        This method computes the Sliced Wasserstein distance between diag1 and diag2

        :param diag1: persistence diagram 1
        :param diag2: second persistence diagram

        :return: a vector with the respective Sliced wasserstein distance
        per homology group dimension

        Sliced Wasserstein Kernels for persistence diagrams were introduced
        by Carriere et al, 2017 and implemented by Alice Patania.
        '''
        diag1, diag2 = PersistenceDiagramHelper.equalize_dimensions(diag1, diag2)
        max_dim = len(diag1)
        result = []
        for d in range(max_dim):
            dist = persim.sliced_wasserstein(np.array(diag1[d]), np.array(diag2[d]), M=slice)
            # dist = DistanceMatrixHelper.compute_swd_between_points(np.array(diag1[d]), np.array(diag2[d]))
            result.append(dist)

        return result

    @staticmethod
    def compute_pd_tree_swd_distance(diag1, diag2, p=2):
        '''
        Implementation of tree
        '''
        pass

    @staticmethod
    def is_common_type(dtype):
        return dtype not in (DistanceMatrixHelper.ETDA_H1,
                             DistanceMatrixHelper.ETDA,
                             DistanceMatrixHelper.ETDA_METRIC,
                             DistanceMatrixHelper.ETDA_METRIC_H1,
                             DistanceMatrixHelper.NP_ETDA_H1,
                             DistanceMatrixHelper.NP_ETDA,
                             DistanceMatrixHelper.NP_ETDA_METRIC,
                             DistanceMatrixHelper.NP_ETDA_METRIC_H1,
                             DistanceMatrixHelper.SWD_H1)

    @staticmethod
    def is_swd_type(dtype):
        return dtype in (DistanceMatrixHelper.SWD,
                             DistanceMatrixHelper.SWD_H1)

    def is_h1_distance(self, dtype):
        return dtype in (DistanceMatrixHelper.BASIC_ETD_H1,
                             DistanceMatrixHelper.ETDA_H1,
                             DistanceMatrixHelper.ETDA_METRIC_H1,
                             DistanceMatrixHelper.HERA_WD_H1,
                             DistanceMatrixHelper.SWD_H1,
                             DistanceMatrixHelper.FISHER_H1,
                             DistanceMatrixHelper.PS_H1
                         )

    @staticmethod
    def compute_distance_to_diagonal(dtype, diag1, p=2, angle_set = None):
        '''
        This method computes the distance to diagonal

        :param diag1: persistence diagram 1
        :param p: the internal power, 2 by default in the future it will be used
        :param angle_set: collection of items representing angle_sets or slices

        :return: a vector with the respective distance to diagonal per Hd
        '''

        diagonal_diag = DistanceMatrixHelper.compute_diagonal_diagram(diag1)
        if DistanceMatrixHelper.is_common_type(dtype):
            d2d = DistanceMatrixHelper.distance_functions[dtype](diag1,
                                                                 diagonal_diag,
                                                                 p)
        else:
            d2d = {angle: 0 for angle in angle_set}
            for angle in angle_set:
                tmp_d2d = DistanceMatrixHelper.distance_functions[dtype](diag1,
                                                                     diagonal_diag,
                                                                     p,
                                                                     angle_set[angle])
                d2d[angle] = tmp_d2d

        return d2d

    @staticmethod
    def compute_diagonal_diagram(diag1):
        '''
        This method computes the diagonal diag

        :param diag1: persistence diagram 1

        :return: a diagonal diag corresponding to diag1
        '''

        return PersistenceDiagramHelper.get_diagonal_diagram(diag1)

    @staticmethod
    def compute_pd_wd_distance(diag1, diag2, p=2):
        '''
        This method computes the pWasserstein distance between diag1 and diag2

        :param diag1: persistence diagram 1
        :param diag2: second persistence diagram
        :param p: the internal power, 2 by default in the future it will be used

        :return: a vector with the respective pwasserstein distance
        per homology group dimension
        '''
        diag1, diag2 = PersistenceDiagramHelper.equalize_dimensions(diag1, diag2)
        max_dim = len(diag1)
        result = []
        for d in range(max_dim):
            dist = persim.wasserstein(np.array(diag1[d]), np.array(diag2[d]))
            result.append(dist)

        return result

    @staticmethod
    def compute_pd_herawd_distance(diag1, diag2, p=2, q = 2):
        '''
        This method computes the pWasserstein distance between diag1 and diag2

        :param diag1: persistence diagram 1
        :param diag2: second persistence diagram
        :param p: the internal power, 2 by default in the future it will be used

        :return: a vector with the respective pwasserstein distance
        per homology group dimension
        '''
        diag1, diag2 = PersistenceDiagramHelper.equalize_dimensions(diag1, diag2)
        max_dim = len(diag1)
        result = []
        for d in range(max_dim):
            dist = hera_wd(np.array(diag1[d]), np.array(diag2[d]), order=q, internal_p=p)
            result.append(dist)

        return result

    @staticmethod
    def compute_swd_between_points(point_cloud1, point_cloud2):
        wd = ot.sliced_wasserstein_distance(point_cloud1, point_cloud2, seed=0)
        return wd

    @staticmethod
    def compute_wd_between_points(point_cloud1, point_cloud2):
        wd = ot.wasserstein_1d(point_cloud1, point_cloud2, p=2, require_sort=True)
        return np.linalg.norm(wd)

    @staticmethod
    def compute_pd_etda_distance(diag1, diag2, p, angle):
        '''
        This method computes the Extended Topology distance for a given set of angles
        :param diag1: PD1
        :param diag2: PD2
        :param p: interval value of the Minkowsky distance
        :param angle:
        :return:
        '''
        return ExtendedTopologyDistanceHelper.get_etd_alpha(diag1, diag2, p, angle)

    @staticmethod
    def compute_pd_etda_with_metric(diag1, diag2, metric, angle=None):
        return ExtendedTopologyDistanceHelper.get_etd_alpha_metric(diag1, diag2, metric, angle)

    @staticmethod
    def compute_pd_basic_etd_distance(diag1, diag2, p=2):
        return ExtendedTopologyDistanceHelper.get_basic_etd(diag1, diag2, p)

    @staticmethod
    def compute_pd_persistent_statistics(diag1, diag2, p):
        '''
        For PD^j(X) = {(bi, di) | i \in I_j} PS includes quantile, average,
        and variance statistics about the following nonnegative numbers

        {bi}_i, {d_i}_i, {(b_i + d_i)/2}_i, {d_i-b_i}_i

        :param diag1:
        :param diag2:
        :param p:
        :return:
        '''
        diag1, diag2 = PersistenceDiagramHelper.equalize_dimensions(diag1, diag2)
        max_dim = len(diag1)
        PS_dist = []

        for d in range(max_dim):
            ps1 = DistanceMatrixHelper.compute_persistent_statistic(diag1[d])
            ps2 = DistanceMatrixHelper.compute_persistent_statistic(diag2[d])
            dist = (np.linalg.norm(np.subtract(ps1, ps2), ord=p))
            PS_dist.append(dist)

        return PS_dist

    '''
    Adapted version from the GetPersStats method from https://github.com/dashtiali/vectorisation-app
    '''
    @staticmethod
    def compute_persistent_statistic( diag):
        barcode = np.array(diag)
        sumb = np.sum(barcode)
        if (np.size(diag) > 0) and sumb > 0:
            # Average of Birth and Death of the barcode
            bc_av0, bc_av1 = np.mean(barcode, axis=0)
            # STDev of Birth and Death of the barcode
            bc_std0, bc_std1 = np.std(barcode, axis=0)
            # Median of Birth and Death of the barcode
            bc_med0, bc_med1 = np.median(barcode, axis=0)
            # Intercuartil range of births and death
            bc_iqr0, bc_iqr1 = np.subtract(*np.percentile(barcode, [75, 25], axis=0))
            # Range of births and deaths
            bc_r0, bc_r1 = np.max(barcode, axis=0) - np.min(barcode, axis=0)
            # Percentiles of births and deaths
            bc_p10_0, bc_p10_1 = np.percentile(barcode, 10, axis=0)
            bc_p25_0, bc_p25_1 = np.percentile(barcode, 25, axis=0)
            bc_p75_0, bc_p75_1 = np.percentile(barcode, 75, axis=0)
            bc_p90_0, bc_p90_1 = np.percentile(barcode, 90, axis=0)

            avg_barcodes = (barcode[:, 1] + barcode[:, 0]) / 2
            # Average of midpoints of the barcode
            bc_av_av = np.mean(avg_barcodes)
            # STDev of midpoints of the barcode
            bc_std_av = np.std(avg_barcodes)
            # Median of midpoints of the barcode
            bc_med_av = np.median(avg_barcodes)
            # Intercuartil range of midpoints
            bc_iqr_av = np.subtract(*np.percentile(avg_barcodes, [75, 25]))
            # Range of midpoints
            bc_r_av = np.max(avg_barcodes) - np.min(avg_barcodes)
            # Percentiles of midpoints
            bc_p10_av = np.percentile(barcode, 10)
            bc_p25_av = np.percentile(barcode, 25)
            bc_p75_av = np.percentile(barcode, 75)
            bc_p90_av = np.percentile(barcode, 90)

            diff_barcode = np.subtract([i[1] for i in barcode], [
                i[0] for i in barcode])
            diff_barcode = np.absolute(diff_barcode)
            # Average of the length of Bars
            bc_lengthAverage = np.mean(diff_barcode)
            # STD of length of Bars
            bc_lengthSTD = np.std(diff_barcode)
            # Median of length of Bars
            bc_lengthMedian = np.median(diff_barcode)
            # Intercuartil range of length of the bars
            bc_lengthIQR = np.subtract(*np.percentile(diff_barcode, [75, 25]))
            # Range of length of the bars
            bc_lengthR = np.max(diff_barcode) - np.min(diff_barcode)
            # Percentiles of lengths of the bars
            bc_lengthp10 = np.percentile(diff_barcode, 10)
            bc_lengthp25 = np.percentile(diff_barcode, 25)
            bc_lengthp75 = np.percentile(diff_barcode, 75)
            bc_lengthp90 = np.percentile(diff_barcode, 90)

            # Number of Bars
            bc_count = len(diff_barcode)
            # Persitent Entropy
            ent = Entropy()

            bc_ent = ent.fit_transform([barcode])
            bar_stats = np.array([bc_av0, bc_av1, bc_std0, bc_std1, bc_med0, bc_med1,
                                  bc_iqr0, bc_iqr1, bc_r0, bc_r1, bc_p10_0, bc_p10_1,
                                  bc_p25_0, bc_p25_1, bc_p75_0, bc_p75_1, bc_p90_0,
                                  bc_p90_1,
                                  bc_av_av, bc_std_av, bc_med_av, bc_iqr_av, bc_r_av, bc_p10_av,
                                  bc_p25_av, bc_p75_av, bc_p90_av, bc_lengthAverage, bc_lengthSTD,
                                  bc_lengthMedian, bc_lengthIQR, bc_lengthR, bc_lengthp10,
                                  bc_lengthp25, bc_lengthp75, bc_lengthp90, bc_count,
                                  bc_ent[0][0]])

        else:
            bar_stats = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0])

        bar_stats[~np.isfinite(bar_stats)] = 0

        return bar_stats

    @staticmethod
    def compute_pfisher_kernel(dia1, diag2, p=2):
        '''
        Persistence Fisher Kernel is implemented in gudhi

        This computes the fisher kernel matrix from a list of persistence diagrams.
        It is computed by exponentiating the corresponding persistence Fisher distance with
        a Gaussian Kernel. See
        papers.nips.cc/paper/8205-persistence-fisher-kernel-a-riemannian-manifold-kernel-for-persistence-diagrams
        for more details.

        '''
        dia1, diag2 = PersistenceDiagramHelper.equalize_dimensions(dia1, diag2)
        max_dim = len(dia1)
        pf = []

        for q in range(max_dim):
            tmp_pf = gudhi.representations.pairwise_persistence_diagram_distances(X=np.array([dia1[q]]),
                                                                                  Y=np.array([diag2[q]]),
                                                                                  metric="persistence_fisher")
            # PF = gudhi.representations.PersistenceFisherKernel(bandwidth_fisher=.001,bandwidth=.001, kernel_approx=None)
            # PF.fit([dia1[q]])
            # tmp_pf = PF.transform([diag2[q]])
            #
            pf.append(tmp_pf[0][0])
        return pf

    @staticmethod
    def compute_pscalespace_kernel(dia1, diag2, p):
        pass
    '''
    
    '''

    @staticmethod
    def compute_pd_swd_distance_h1(diag1, diag2, p=2, slice=50):
        '''
        This method computes the Sliced Wasserstein distance between diag1 and diag2

        :param diag1: persistence diagram 1
        :param diag2: second persistence diagram

        :return: a vector with the respective Sliced wasserstein distance
        per homology group dimension

        Sliced Wasserstein Kernels for persistence diagrams were introduced
        by Carriere et al, 2017 and implemented by Alice Patania.
        '''
        diag1, diag2 = PersistenceDiagramHelper.equalize_dimensions(diag1, diag2)
        dist = persim.sliced_wasserstein(np.array(diag1[1]), np.array(diag2[1]), M=slice)

        return dist

    @staticmethod
    def compute_pd_tree_swd_distance_h1(diag1, diag2, p=2):
        '''
        Implementation of tree
        '''
        pass

    @staticmethod
    def compute_pd_wd_distance_h1(diag1, diag2, p=2):
        '''
        This method computes the pWasserstein distance between diag1 and diag2

        :param diag1: persistence diagram 1
        :param diag2: second persistence diagram
        :param p: the internal power, 2 by default in the future it will be used

        :return: a vector with the respective pwasserstein distance
        per homology group dimension
        '''
        diag1, diag2 = PersistenceDiagramHelper.equalize_dimensions(diag1, diag2)
        dist = persim.wasserstein(np.array(diag1[1]), np.array(diag2[1]))

        return dist

    @staticmethod
    def compute_pd_herawd_distance_h1(diag1, diag2, p=2, q=2):
        '''
        This method computes the pWasserstein distance between diag1 and diag2

        :param diag1: persistence diagram 1
        :param diag2: second persistence diagram
        :param p: the internal power, 2 by default in the future it will be used

        :return: a vector with the respective pwasserstein distance
        per homology group dimension
        '''
        diag1, diag2 = PersistenceDiagramHelper.equalize_dimensions(diag1, diag2)
        dist = hera_wd(np.array(diag1[1]), np.array(diag2[1]), order=q, internal_p=p)

        return dist

    @staticmethod
    def compute_pd_etda_distance_h1(diag1, diag2, p, angle):
        '''
        This method computes the Extended Topology distance for a given set of angles
        :param diag1: PD1
        :param diag2: PD2
        :param p: interval value of the Minkowsky distance
        :param angle:
        :return:
        '''
        return ExtendedTopologyDistanceHelper.get_etd_alpha_in_H1(diag1, diag2, p, angle)

    @staticmethod
    def compute_pd_etda_with_metric_h1(diag1, diag2, metric, angle=None):
        return ExtendedTopologyDistanceHelper.get_etd_alpha_metric_in_H1(diag1, diag2, metric, angle)

    @staticmethod
    def compute_pd_basic_etd_distance_h1(diag1, diag2, p=2):
        return ExtendedTopologyDistanceHelper.get_basic_etd_in_H1(diag1, diag2, p)

    @staticmethod
    def compute_pd_persistent_statistics_h1(diag1, diag2, p):
        '''
        For PD^j(X) = {(bi, di) | i \in I_j} PS includes quantile, average,
        and variance statistics about the following nonnegative numbers

        {bi}_i, {d_i}_i, {(b_i + d_i)/2}_i, {d_i-b_i}_i

        :param diag1:
        :param diag2:
        :param p:
        :return:
        '''
        diag1, diag2 = PersistenceDiagramHelper.equalize_dimensions(diag1, diag2)

        ps1 = DistanceMatrixHelper.compute_persistent_statistic(diag1[1])
        ps2 = DistanceMatrixHelper.compute_persistent_statistic(diag2[1])
        dist = np.linalg.norm(np.subtract(ps1, ps2), ord=p)

        return dist

    '''
    Adapted version from the GetPersStats method from https://github.com/dashtiali/vectorisation-app
    '''

    @staticmethod
    def compute_pfisher_kernel_h1(dia1, diag2, p=2):
        '''
        Persistence Fisher Kernel is implemented in gudhi

        This computes the fisher kernel matrix from a list of persistence diagrams.
        It is computed by exponentiating the corresponding persistence Fisher distance with
        a Gaussian Kernel. See
        papers.nips.cc/paper/8205-persistence-fisher-kernel-a-riemannian-manifold-kernel-for-persistence-diagrams
        for more details.

        '''
        dia1, diag2 = PersistenceDiagramHelper.equalize_dimensions(dia1, diag2)
        tmp_pf = gudhi.representations.pairwise_persistence_diagram_distances(X=np.array([dia1[1]]),
                                                                              Y=np.array([diag2[1]]),
                                                                              metric="persistence_fisher")

        return tmp_pf[0][0]

    @staticmethod
    def compute_np_etda_distance(diag1, diag2, p, angle):
        '''
        This method computes the Extended Topology distance for a given set of angles
        :param diag1: PD1
        :param diag2: PD2
        :param p: interval value of the Minkowsky distance
        :param angle:
        :return:
        '''
        return FastExtendedTopologyDistanceHelper.get_etd_alpha_new(diag1, diag2, p, angle)

    @staticmethod
    def compute_np_etda_with_metric(diag1, diag2, metric, angle=None):
        return FastExtendedTopologyDistanceHelper.get_etd_alpha_metric_new(diag1, diag2, metric, angle)

    @staticmethod
    def compute_np_basic_etd_distance(diag1, diag2, p=2):
        return FastExtendedTopologyDistanceHelper.get_basic_etd_new(diag1, diag2, p)

    @staticmethod
    def compute_np_etda_distance_h1(diag1, diag2, p, angle):
        '''
        This method computes the Extended Topology distance for a given set of angles
        :param diag1: PD1
        :param diag2: PD2
        :param p: interval value of the Minkowsky distance
        :param angle:
        :return:
        '''
        return ExtendedTopologyDistanceHelper.get_etd_alpha_in_H1_new(diag1, diag2, p, angle)

    @staticmethod
    def compute_np_etda_with_metric_h1(diag1, diag2, metric, angle=None):
        return ExtendedTopologyDistanceHelper.get_etd_alpha_metric_in_H1_new(diag1, diag2, metric, angle)

    @staticmethod
    def compute_np_basic_etd_distance_h1(diag1, diag2, p=2):
        return ExtendedTopologyDistanceHelper.get_basic_etd_in_H1_new(diag1, diag2, p)
    '''

    '''

    distance_functions = {BASIC_ETD: compute_pd_basic_etd_distance,
                          ETDA: compute_pd_etda_distance,
                          ETDA_METRIC: compute_pd_etda_with_metric,
                          WD: compute_pd_wd_distance,
                          HERA_WD: compute_pd_herawd_distance,
                          SWD: compute_pd_swd_distance,
                          PS: compute_pd_persistent_statistics,
                          FISHER: compute_pfisher_kernel,
                          BASIC_ETD_H1: compute_pd_basic_etd_distance_h1,
                          ETDA_H1: compute_pd_etda_distance_h1,
                          ETDA_METRIC_H1: compute_pd_etda_with_metric_h1,
                          NP_BASIC_ETD_H1: compute_np_basic_etd_distance_h1,
                          NP_BASIC_ETD: compute_np_basic_etd_distance,
                          NP_ETDA: compute_np_etda_distance,
                          NP_ETDA_H1: compute_np_etda_distance_h1,
                          NP_ETDA_METRIC: compute_np_etda_with_metric,
                          NP_ETDA_METRIC_H1: compute_np_etda_with_metric_h1,
                          WD_H1: compute_pd_wd_distance_h1,
                          HERA_WD_H1: compute_pd_herawd_distance_h1,
                          SWD_H1: compute_pd_swd_distance_h1,
                          PS_H1: compute_pd_persistent_statistics_h1,
                          FISHER_H1: compute_pfisher_kernel_h1}


if __name__ == '__main__':
    from utilities.directoy_helper import DirectoryHelper
    from utilities.tda_helper import TDAHelper
    outpath = "results/SupervisedLearningApp/Shrec07Processor/topological_info"
    outpath_dst = "results/SupervisedLearningApp/Shrec07Processor/distance_folder"
    all_names = DirectoryHelper.get_all_filenames(root_path=outpath, file_pattern="diag.pickle")
    all_names = DirectoryHelper.sort_filenames_by_suffix(all_names, sep="_")
    all_pds = TDAHelper.get_all_pdiagrams(all_names, None)

    distance, avgtime =DistanceMatrixHelper.compute_pd_distance_matrix_with_diagrams(distance_type=DistanceMatrixHelper.BASIC_ETD,
                                                                  all_pds=all_pds, p=2, angle_set={},
                                   distance_folder=outpath_dst, size=len(all_names))

    print("hello")

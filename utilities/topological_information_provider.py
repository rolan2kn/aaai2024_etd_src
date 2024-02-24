# -*- coding: utf-8 -*-

import os
import time
from gudhi.representations import Entropy

from utilities.distance_matrix_helper import DistanceMatrixHelper
from utilities.extended_topology_distance import ExtendedTopologyDistanceHelper
from utilities.persistence_diagram_helper import PersistenceDiagramHelper
from utilities.information_visualization_helpers import InformationVisualizationHelper


class TopologicalInformationHelper:
    CONTROL, TARGET_DIV, DIV_TARGET, LOG_TARGET, LOG_DIV_TARGET, LOG_DIV_WD = range(6)
    time_in_seconds = {}

    @staticmethod
    def get_persistence_entropy_curves(full_data):
        max_dim = PersistenceDiagramHelper.get_maximal_dimension(full_data)

        persistence_entropy_labels = []
        persistence_entropy_labels.extend(
            ["PE" for _ in range(max_dim)])
        TopologicalInformationHelper.time_in_seconds["PE"] = 0
        persistence_entropy = []

        for diagrams in full_data:  ## compute persistent entropy

            entropy_values = []
            t1 = time.time_ns()
            Y = Entropy().fit_transform(X=np.array(diagrams, dtype=object))
            entropy_values.extend([y[0] for y in Y])
            TopologicalInformationHelper.time_in_seconds["PE"] += (time.time_ns() - t1) / 10**6
            persistence_entropy.append(entropy_values)

        TopologicalInformationHelper.time_in_seconds["PE"] /= len(full_data)

        return persistence_entropy, persistence_entropy_labels

    @staticmethod
    def get_persistence_statistic_curves(full_data, approach=2):
        max_dim = PersistenceDiagramHelper.get_maximal_dimension(full_data)

        ps_labels = []
        ps_labels.extend(
            ["PS" for d in range(max_dim)])

        TopologicalInformationHelper.time_in_seconds["PS"] = 0
        ps_all = []
        prev_wd = None
        for diagrams in full_data:  ## compute persistent entropy
            if prev_wd is None:
                prev_wd = diagrams

            t1 = time.time_ns()
            Y = DistanceMatrixHelper.compute_pd_persistent_statistics(prev_wd, diagrams, p=2)
            TopologicalInformationHelper.time_in_seconds["PS"] += (time.time_ns() - t1) / 10 ** 6
            ps_all.append(Y)
            if approach == 1:
                prev_wd = diagrams

        TopologicalInformationHelper.time_in_seconds["PS"] /= len(full_data)

        return ps_all, ps_labels

    @staticmethod
    def get_etda_distance_curves(full_data, p = 2, angle_set={}, approach=2):
        asize = len(angle_set)
        max_dim = PersistenceDiagramHelper.get_maximal_dimension(full_data)
        if asize == 0:
            return [], []

        td_labels = []

        for angle in angle_set:
            TopologicalInformationHelper.time_in_seconds.update({f"ETD_{angle}": 0})
            for d in range(max_dim):
                td_labels.append(f"ETDA_{angle}")

        td_collection = []
        prev = None
        for diagrams in full_data:  ## compute persistent entropy
            diagrams = PersistenceDiagramHelper.truncate_diagram(PD=diagrams, max_eps=None)

            if prev is None:
                prev = diagrams

            etda_val = []
            for angle in angle_set:
                t1 = time.time_ns()
                etda = DistanceMatrixHelper.compute_pd_etda_distance(prev, diagrams, p, angle_set[angle])
                TopologicalInformationHelper.time_in_seconds[f"ETD_{angle}"] += (time.time_ns() - t1) / 10 ** 6
                etda_val.extend(etda)
            td_collection.append(etda_val)

            if approach == 1:
                prev = diagrams

        for angle in angle_set:
            TopologicalInformationHelper.time_in_seconds[f"ETD_{angle}"] /= len(full_data)

        return td_collection, td_labels

    @staticmethod
    def get_basic_etd_distance_curves(full_data, p=2, approach=2):
        max_dim = PersistenceDiagramHelper.get_maximal_dimension(full_data)
        etd_labels = []
        for d in range(max_dim):
            etd_labels.append(f"BasicETD")
        TopologicalInformationHelper.time_in_seconds["BasicETD"] = 0
        td_collection = []
        prev = None
        for diagrams in full_data:  ## compute persistent entropy
            if prev is None:
                prev = diagrams

            t1 = time.time_ns()
            etd = DistanceMatrixHelper.compute_pd_basic_etd_distance(prev, diagrams, p)
            TopologicalInformationHelper.time_in_seconds["BasicETD"] += (time.time_ns() - t1) / 10 ** 6
            td_collection.append(etd)

            if approach == 1:
                prev = diagrams
        TopologicalInformationHelper.time_in_seconds["BasicETD"] /= len(full_data)
        return td_collection, etd_labels

    @staticmethod
    def get_np_etda_distance_curves(full_data,
                                    p = 2,
                                    angle_set={},
                                    approach=2):
        asize = len(angle_set)
        max_dim = PersistenceDiagramHelper.get_maximal_dimension(full_data)
        if asize == 0:
            return [], []

        td_labels = []

        for angle in angle_set:
            TopologicalInformationHelper.time_in_seconds.update({f"npETD_{angle}": 0})
            for d in range(max_dim):
                td_labels.append(f"npETDA_{angle}")

        td_collection = []
        prev = None
        for diagrams in full_data:  ## compute persistent entropy
            diagrams = PersistenceDiagramHelper.truncate_diagram(PD=diagrams, max_eps=None)

            if prev is None:
                prev = diagrams

            etda_val = []
            for angle in angle_set:
                t1 = time.time_ns()
                etda = DistanceMatrixHelper.compute_np_etda_distance(prev, diagrams, p, angle_set[angle])
                TopologicalInformationHelper.time_in_seconds[f"npETD_{angle}"] += (time.time_ns() - t1) / 10 ** 6
                etda_val.extend(etda)
            td_collection.append(etda_val)

            if approach == 1:
                prev = diagrams

        for angle in angle_set:
            TopologicalInformationHelper.time_in_seconds[f"npETD_{angle}"] /= len(full_data)

        return td_collection, td_labels

    @staticmethod
    def get_np_basic_etd_distance_curves(full_data, p=2, approach=2):
        max_dim = PersistenceDiagramHelper.get_maximal_dimension(full_data)
        etd_labels = []
        for d in range(max_dim):
            etd_labels.append(f"npBasicETD")
        TopologicalInformationHelper.time_in_seconds["npBasicETD"] = 0
        td_collection = []
        prev = None
        for diagrams in full_data:  ## compute persistent entropy
            if prev is None:
                prev = diagrams

            t1 = time.time_ns()
            etd = DistanceMatrixHelper.compute_pd_basic_etd_distance(prev, diagrams, p)
            TopologicalInformationHelper.time_in_seconds["npBasicETD"] += (time.time_ns() - t1) / 10 ** 6
            td_collection.append(etd)

            if approach == 1:
                prev = diagrams
        TopologicalInformationHelper.time_in_seconds["npBasicETD"] /= len(full_data)
        return td_collection, etd_labels
    @staticmethod
    def get_sliced_wasserstein_curves(full_data, approach=2):
        """
        Compute the Sliced Wasserstein distance curves. We use Persim Library to perform distance computations.
        The idea is to compute distances following two approaches:
        
        1. Between each two consecutive layers: This could be useful to understand local (between layers) topological
         changes.
        2. Between any layer with the first one: This is meaningful to understand global topological changes per layer.

        By default we return the second approach.

        :param full_data: dictionary with persistence information per layer
        :param activation_function: name of activation function
        :param approach: this method is to choose the desired comparison approach.
        :return:
        """
        hgroups = PersistenceDiagramHelper.get_maximal_dimension(full_data)

        wdc_labels = []
        wdc_labels.extend(
            ["SWD" for d in range(hgroups)])

        wdc = []
        TopologicalInformationHelper.time_in_seconds[f"SWD"] = 0
        prev_wd = None
        for diagrams in full_data:  ## compute persistent entropy

            if prev_wd is None:
                prev_wd = diagrams

            t1 = time.time_ns()
            wd_values = DistanceMatrixHelper.compute_pd_swd_distance(prev_wd, diagrams)
            TopologicalInformationHelper.time_in_seconds["SWD"] += (time.time_ns() - t1) / 10 ** 6

            wdc.append(wd_values)
            if approach == 1:
                prev_wd = diagrams

        TopologicalInformationHelper.time_in_seconds["SWD"] /= len(full_data)

        return wdc, wdc_labels

    @staticmethod
    def get_persim_swd_distance_curves(full_data, slices=[50], approach=2):
        asize = len(slices)
        max_dim = PersistenceDiagramHelper.get_maximal_dimension(full_data)
        if asize == 0:
            return [], []

        swd_labels = []

        for M in slices:
            TopologicalInformationHelper.time_in_seconds.update({f"SWD_M{M}": 0})
            for d in range(max_dim):
                swd_labels.append(f"SWD_M{M}")

        swd_collection = []
        prev = None
        for diagrams in full_data:  ## compute persistent entropy

            if prev is None:
                prev = diagrams

            swd_val = []

            for M in slices:
                t1 = time.time_ns()
                wd_values = DistanceMatrixHelper.compute_pd_swd_distance(prev, diagrams, slice=M)
                TopologicalInformationHelper.time_in_seconds[f"SWD_M{M}"] += (time.time_ns() - t1) / 10 ** 6
                swd_val.extend(wd_values)
            swd_collection.append(swd_val)

            if approach == 1:
                prev = diagrams

        for M in slices:
            TopologicalInformationHelper.time_in_seconds[f"SWD_M{M}"] /= len(full_data)

        return swd_collection, swd_labels

    @staticmethod
    def get_p_wasserstein_curves(full_data, approach=2):
        """
        Compute the Wasserstein distance curves. We use Persim Library to compute Wasserstein distances.
        The idea is to compute distances following two approaches:

        1. Between each two consecutive layers: This could be useful to understand local (between layers) topological
         changes.
        2. Between any layer with the first one: This is meaningful to understand global topological changes per layer.

        By default we return the second approach.

        :param full_data: dictionary with persistence information per layer
        :param activation_function: name of activation function
        :param approach: this method is to choose the desired comparison approach.
        :return:
        """
        hgroups = PersistenceDiagramHelper.get_maximal_dimension(full_data)

        wdc_labels = []
        wdc_labels.extend(
            ["WD" for d in range(hgroups)])
        TopologicalInformationHelper.time_in_seconds["WD"] = 0
        wdc = []
        prev_wd = None
        for diagrams in full_data:  ## compute persistent entropy

            if prev_wd is None:
                prev_wd = diagrams

            t1 = time.time_ns()
            wd_values = DistanceMatrixHelper.compute_pd_wd_distance(prev_wd, diagrams)
            TopologicalInformationHelper.time_in_seconds["WD"] += (time.time_ns() - t1) / 10 ** 6
            wdc.append(wd_values)

            if approach == 1:
                prev_wd = diagrams

        TopologicalInformationHelper.time_in_seconds["WD"] /= len(full_data)
        return wdc, wdc_labels

    @staticmethod
    def get_hera_wasserstein_curves(full_data, approach=2):
        """
        Compute the Wasserstein distance curves. We use Persim Library to compute Wasserstein distances.
        The idea is to compute distances following two approaches:

        1. Between each two consecutive layers: This could be useful to understand local (between layers) topological
         changes.
        2. Between any layer with the first one: This is meaningful to understand global topological changes per layer.

        By default we return the second approach.

        :param full_data: dictionary with persistence information per layer
        :param activation_function: name of activation function
        :param approach: this method is to choose the desired comparison approach.
        :return:
        """
        hgroups = PersistenceDiagramHelper.get_maximal_dimension(full_data)

        wdc_labels = []
        wdc_labels.extend(
            ["HeraWD" for d in range(hgroups)])
        TopologicalInformationHelper.time_in_seconds["HERA_WD"] = 0
        wdc = []
        prev_wd = None
        for diagrams in full_data:  ## compute persistent entropy

            if prev_wd is None:
                prev_wd = diagrams

            t1 = time.time_ns()
            wd_values = DistanceMatrixHelper.compute_pd_herawd_distance(prev_wd, diagrams)
            TopologicalInformationHelper.time_in_seconds["HERA_WD"] += (time.time_ns() - t1) / 10 ** 6
            wdc.append(wd_values)

            if approach == 1:
                prev_wd = diagrams

        TopologicalInformationHelper.time_in_seconds["HERA_WD"] /= len(full_data)
        return wdc, wdc_labels

    @staticmethod
    def get_fisher_kernel_curves(full_data, approach=2):
        """
        Compute the Fisher KErnel curves. We use Gudhi Library to compute them.
        The idea is to compute distances following two approaches:

        1. Between each two consecutive layers: This could be useful to understand local (between layers) topological
         changes.
        2. Between any layer with the first one: This is meaningful to understand global topological changes per layer.

        By default we return the second approach.

        :param full_data: dictionary with persistence information per layer
        :param activation_function: name of activation function
        :param approach: this method is to choose the desired comparison approach.
        :return:
        """
        hgroups = PersistenceDiagramHelper.get_maximal_dimension(full_data)

        fkc_labels = []
        fkc_labels.extend(
            ["FisherKernel" for d in range(hgroups)])
        TopologicalInformationHelper.time_in_seconds["FISHER_KERNEL"] = 0
        fkc = []
        prev_fk = None
        for diagrams in full_data:  ## compute persistent entropy

            if prev_fk is None:
                prev_fk = diagrams

            t1 = time.time_ns()
            fk_values = DistanceMatrixHelper.compute_pfisher_kernel(prev_fk, diagrams)
            TopologicalInformationHelper.time_in_seconds["FISHER_KERNEL"] += (time.time_ns() - t1) / 10 ** 6
            fkc.append(fk_values)

            if approach == 1:
                prev_fk = diagrams

        TopologicalInformationHelper.time_in_seconds["FISHER_KERNEL"] /= len(full_data)
        return fkc, fkc_labels

    @staticmethod
    def draw_persistence_entropy(full_data, activation_function="RELU", result_folder=None, approach=2):
        """
        Compute and draw persistence entropy curves

        :param full_data: dictionary with persistence information per layer
        :param activation_function: name of activation function
        :return:
        """
        pec, pec_labels = TopologicalInformationHelper.get_persistence_entropy_curves(full_data)
        InformationVisualizationHelper.plot_persistence_curves(pec, pec_labels,
                                                               "Persistence Entropy (PE) curves", "Persistence entropy",
                                activation_function, result_folder, approach)

    @staticmethod
    def draw_persistence_statistics(full_data, activation_function="RELU", result_folder=None, approach=2):
        """
        Compute and draw persistence statistics curves

        :param full_data: dictionary with persistence information per layer
        :param activation_function: name of activation function
        :return:
        """
        pec, pec_labels = TopologicalInformationHelper.get_persistence_statistic_curves(full_data,
                                                                                        activation_function,
                                                                                        approach)
        InformationVisualizationHelper.plot_persistence_curves(pec, pec_labels,
                                                               "Persistence Statistics (PS) curves",
                                                               "PS distance",
                                                               activation_function, result_folder, approach)

    @staticmethod
    def draw_etda_distances(full_data,
                            activation_function="RELU",
                            p=2,
                            angle_set = None,
                            result_folder = None,
                            approach=2):
        """
        Compute the Topological distance curves. We use collect_all_topological_distances.
        The idea is to compute distances following two approaches:

        1. Between each two consecutive layers: This could be useful to understand local (between layers)
        topological
         changes.
        2. Between any layer with the first one: This is meaningful to understand global topological changes
        per layer.

        By default we return the second approach.

        :param full_data: dictionary with persistence information per layer
        :param activation_function: name of activation function
        :param approach: this method is to choose the desired comparison approach.
        :return:
        """

        td_collection, td_labels = TopologicalInformationHelper.get_etda_distance_curves(
                                        full_data, p=p,
                                        angle_set=angle_set, approach=approach)

        InformationVisualizationHelper.plot_persistence_curves(curves=td_collection, labels=td_labels,
                                                               title="ETDA distance curves",
                                                               ylabel="ETDA distances",
                                                              activation_function=activation_function,
                                                               result_folder=result_folder, approach=approach)

    @staticmethod
    def draw_basic_etd_distances(full_data, activation_function="RELU", p=2, result_folder = None, approach=2):
        """
        Compute the Topological distance curves. We use collect_all_topological_distances.
        The idea is to compute distances following two approaches:

        1. Between each two consecutive layers: This could be useful to understand local (between layers)
        topological
         changes.
        2. Between any layer with the first one: This is meaningful to understand global topological changes
        per layer.

        By default we return the second approach.

        :param full_data: dictionary with persistence information per layer
        :param activation_function: name of activation function
        :param approach: this method is to choose the desired comparison approach.
        :return:
        """

        td_collection, td_labels = TopologicalInformationHelper.\
            get_basic_etd_distance_curves(full_data,
                                          p=p, approach=approach)
        InformationVisualizationHelper.plot_persistence_curves(curves=td_collection, labels=td_labels,
                                                               title="Topological distance curve",
                                                                      ylabel="Topological distance",
                                                              activation_function=activation_function,
                                                               result_folder=result_folder, approach=approach)

    @staticmethod
    def draw_np_etda_distances(full_data,
                               activation_function="RELU",
                               p=2,
                               angle_set = None,
                               result_folder = None, approach=2):
        """
        Compute the Topological distance curves. We use collect_all_topological_distances.
        The idea is to compute distances following two approaches:

        1. Between each two consecutive layers: This could be useful to understand local (between layers)
        topological
         changes.
        2. Between any layer with the first one: This is meaningful to understand global topological changes
        per layer.

        By default we return the second approach.

        :param full_data: dictionary with persistence information per layer
        :param activation_function: name of activation function
        :param approach: this method is to choose the desired comparison approach.
        :return:
        """

        td_collection, td_labels = TopologicalInformationHelper.get_np_etda_distance_curves(
            full_data,
            p=p,
            angle_set=angle_set, approach=approach)

        InformationVisualizationHelper.plot_persistence_curves(curves=td_collection, labels=td_labels,
                                                               title="npETDA distance curves",
                                                               ylabel="npETDA distances",
                                                              activation_function=activation_function,
                                                               result_folder=result_folder, approach=approach)

    @staticmethod
    def draw_np_basic_etd_distances(full_data, activation_function="RELU", p=2, result_folder = None, approach=2):
        """
        Compute the Topological distance curves. We use collect_all_topological_distances.
        The idea is to compute distances following two approaches:

        1. Between each two consecutive layers: This could be useful to understand local (between layers)
        topological
         changes.
        2. Between any layer with the first one: This is meaningful to understand global topological changes
        per layer.

        By default we return the second approach.

        :param full_data: dictionary with persistence information per layer
        :param activation_function: name of activation function
        :param approach: this method is to choose the desired comparison approach.
        :return:
        """

        td_collection, td_labels = TopologicalInformationHelper.\
            get_np_basic_etd_distance_curves(full_data,
                                          p=p, approach=approach)
        InformationVisualizationHelper.plot_persistence_curves(curves=td_collection, labels=td_labels,
                                                               title="Topological distance curve",
                                                                      ylabel="Topological distance",
                                                              activation_function=activation_function,
                                                               result_folder=result_folder, approach=approach)

    @staticmethod
    def draw_wasserstein_distances(full_data, activation_function="RELU",p=2,
                                                   result_folder=None, approach=2):
        """
        Computes and draw Wasserstein distance curves

        :param full_data:
        :param activation_function:
        :return:
        """
        wdc, wdc_labels = TopologicalInformationHelper.get_p_wasserstein_curves(full_data,
                                                                                activation_function=activation_function,
                                                                                approach=approach)
        InformationVisualizationHelper.plot_persistence_curves(wdc, wdc_labels,
                                                               "Wasserstein distance curves",
                                                               "Wasserstein distance",
                                activation_function, result_folder, approach=approach)

    @staticmethod
    def draw_fisher_kernels(full_data, activation_function="RELU",
                                                   result_folder=None, approach=2):
        """
        Computes and draw Wasserstein distance curves

        :param full_data:
        :param activation_function:
        :return:
        """
        fkc, fkc_labels = TopologicalInformationHelper.get_fisher_kernel_curves(full_data,
                                                                                   activation_function=activation_function,
                                                                                   approach=approach)
        InformationVisualizationHelper.plot_persistence_curves(fkc, fkc_labels,
                                                               "Fisher Kernel curves",
                                                               "FIsher kernel",
                                                               activation_function, result_folder, approach=approach)

    @staticmethod
    def draw_hera_wasserstein_distances(full_data, activation_function="RELU",
                                                   result_folder=None, approach=2):
        """
        Computes and draw Wasserstein distance curves

        :param full_data:
        :param activation_function:
        :return:
        """
        wdc, wdc_labels = TopologicalInformationHelper.get_hera_wasserstein_curves(full_data,
                                                                                activation_function=activation_function,
                                                                                approach=approach)
        InformationVisualizationHelper.plot_persistence_curves(wdc, wdc_labels,
                                                               "Wasserstein distance curves",
                                                               "Wasserstein distance",
                                activation_function, result_folder, approach=approach)
    @staticmethod
    def draw_sliced_wasserstein_distances(full_data, activation_function="RELU",
                                                   result_folder=None, approach=2):
        """
        Computes and draw Wasserstein distance curves

        :param full_data:
        :param activation_function:
        :return:
        """
        wdc, wdc_labels = TopologicalInformationHelper.get_sliced_wasserstein_curves(full_data,
                                                                              activation_function=activation_function)
        InformationVisualizationHelper.plot_persistence_curves(wdc, wdc_labels, "Default SWD curves",
                                                               "Default SWD",
                                                               activation_function, result_folder, approach=approach)

    @staticmethod
    def draw_persim_swd_distances(full_data, activation_function="RELU",
                                          result_folder=None, slices = [50], approach=2):
        """
        Computes and draw Slices Wasserstein distance curves according to persim and different slides

        :param full_data:
        :param activation_function:
        :return:
        """
        swd_collection, swd_labels = TopologicalInformationHelper.get_persim_swd_distance_curves(full_data,
                                                                         activation_function=activation_function,
                                                                                                 slices=slices,
                                                                                                 approach=approach)
        InformationVisualizationHelper.plot_persistence_curves(swd_collection, swd_labels,
                                                               "SWD distance curves",
                                                               "SWD",
                                                               activation_function, result_folder,
                                                               approach=approach)

    @staticmethod
    def compute_time_of_topo_curves(full_data, activation_function = "RELU", result_folder=None,
                                    angle_set = {}, approach=2):
        file_name = time.strftime(
            f"{result_folder}/%y.%m.%d__%H.%M.%S__avgtime_results_{activation_function}.txt")
        file = open(file_name, "w")
        file.write(f"Activation: {activation_function}\n")
        file.write(f"APPROACH: {approach} 1. all vs first, 2. each vs previous\n")

        TopologicalInformationHelper.draw_persistence_entropy(full_data=full_data,
                                                              activation_function=activation_function,
                                                              result_folder=result_folder,
                                                              approach=approach)

        TopologicalInformationHelper.draw_persistence_statistics(full_data=full_data,
                                                              activation_function=activation_function,
                                                              result_folder=result_folder,
                                                              approach=approach)
        #
        TopologicalInformationHelper.draw_wasserstein_distances(full_data=full_data,
                                                              activation_function=activation_function,
                                                              result_folder=result_folder,
                                                              approach=approach)

        TopologicalInformationHelper.draw_hera_wasserstein_distances(full_data=full_data,
                                                              activation_function=activation_function,
                                                              result_folder=result_folder,
                                                              approach=approach)

        TopologicalInformationHelper.draw_sliced_wasserstein_distances(full_data=full_data,
                                                              activation_function=activation_function,
                                                              result_folder=result_folder,
                                                              approach=approach)

        TopologicalInformationHelper.draw_persim_swd_distances(full_data=full_data,
                                                               activation_function=activation_function,
                                                               result_folder=result_folder,
                                                               slices=[1,2,4,8,16],
                                                               approach=approach)

        TopologicalInformationHelper.draw_basic_etd_distances(full_data=full_data,
                                                              activation_function=activation_function,
                                                              result_folder=result_folder,
                                                              approach=approach)

        TopologicalInformationHelper.draw_etda_distances(full_data=full_data,
                                                              activation_function=activation_function,
                                                              angle_set=angle_set,
                                                              result_folder=result_folder,
                                                              approach=approach)

        for key in TopologicalInformationHelper.time_in_seconds:
            file.write(f"{key}: {TopologicalInformationHelper.time_in_seconds[key]} milliseconds\n")
        file.close()

    @staticmethod
    def apply_transform(reference_value, target_value, transform_type = CONTROL):
        if transform_type == TopologicalInformationHelper.CONTROL:
            return target_value

        if transform_type == TopologicalInformationHelper.LOG_TARGET:
            if target_value < 0.0001:
                return 0
            return np.log(target_value)

        if transform_type == TopologicalInformationHelper.TARGET_DIV:
            return target_value / reference_value if reference_value != 0 else 0

        if transform_type == TopologicalInformationHelper.DIV_TARGET:
            return reference_value / target_value if target_value != 0 else 0

        if transform_type == TopologicalInformationHelper.LOG_DIV_TARGET:
            if reference_value <= 0.0001:
                return 0
            return np.log(target_value / reference_value)

        if transform_type == TopologicalInformationHelper.LOG_DIV_WD:
            if target_value <= 0.0001:
                return 0
            return np.log(reference_value / target_value)

        return target_value

    @staticmethod
    def get_transform_name(transform_type=CONTROL):
        fixed_text = "Distance"
        if transform_type == TopologicalInformationHelper.CONTROL:
            return "CONTROL", fixed_text

        if transform_type == TopologicalInformationHelper.LOG_TARGET:
            return "LOG_TARGET", f"log({fixed_text})"

        if transform_type == TopologicalInformationHelper.TARGET_DIV:
            return "TARGET_DIV", f"{fixed_text}/WD"

        if transform_type == TopologicalInformationHelper.DIV_TARGET:
            return "DIV_TARGET", f"WD/{fixed_text}"

        if transform_type == TopologicalInformationHelper.LOG_DIV_TARGET:
            return "LOG_DIV_TARGET", f"log({fixed_text}/WD)"

        if transform_type == TopologicalInformationHelper.LOG_DIV_WD:
            return "LOG_DIV_WD", f"log(WD/{fixed_text})"

        return "CONTROL", fixed_text

    @staticmethod
    def draw_all_topo_curves_splitted_by_hd(full_data, activation_function = "RELU", result_folder=None,
                                    angle_set = {}, approach=2):
        slices = [1, 2, 4, 8, 16]
        file_name = time.strftime(
            f"{result_folder}/%y.%m.%d__%H.%M.%S__avgtime_results_{activation_function}.txt")
        if not os.path.isdir(result_folder):
            os.makedirs(result_folder)

        file = open(file_name, "w")
        file.write(f"Activation: {activation_function}\n")
        file.write(f"APPROACH: ({approach}) -> 1. each vs previous, 2. each vs first\n")

        # pe_curves, pe_labels = TopologicalInformationHelper.get_persistence_entropy_curves(full_data=full_data,
        #                                                       activation_function=activation_function)

        ps_curves, ps_labels = TopologicalInformationHelper.get_persistence_statistic_curves(full_data=full_data,
                                                              approach=approach)

        fk_curves, fk_labels = TopologicalInformationHelper.get_fisher_kernel_curves(full_data=full_data,
                                                                                          approach=approach)


        wd_curves, wd_labels = TopologicalInformationHelper.get_p_wasserstein_curves(full_data=full_data,
                                                              approach=approach)

        hwd_curves, hwd_labels = TopologicalInformationHelper.get_hera_wasserstein_curves(full_data=full_data,
                                                              approach=approach)

        swd_curves, swd_labels = TopologicalInformationHelper.get_sliced_wasserstein_curves(full_data=full_data,
                                                              approach=approach)

        swd_collection, swdc_labels = TopologicalInformationHelper.get_persim_swd_distance_curves(full_data=full_data,
                                                               slices=slices,
                                                               approach=approach)

        etd_curves, etd_labels = TopologicalInformationHelper.get_basic_etd_distance_curves(full_data=full_data,
                                                              approach=approach)

        etda_curves, etda_labels = TopologicalInformationHelper.get_etda_distance_curves(full_data=full_data,
                                                              angle_set=angle_set,
                                                              approach=approach)

        npetd_curves, npetd_labels = TopologicalInformationHelper.get_np_basic_etd_distance_curves(full_data=full_data,
                                                                                            approach=approach)


        npetda_curves, npetda_labels = TopologicalInformationHelper.get_np_etda_distance_curves(full_data=full_data,
                                                                                         angle_set=angle_set,
                                                                                         approach=approach)

        # we consider npBasicETD and npETDA only for timing purposes.
        # Since they returns the same values per distance the curves are exactly the same

        # collect all curves and labels to separate them by Hd
        max_dim = PersistenceDiagramHelper.get_maximal_dimension(full_data)
        nlayers = len(full_data)
        asize = len(angle_set)
        for transform in [TopologicalInformationHelper.CONTROL,
                          TopologicalInformationHelper.TARGET_DIV,
                          TopologicalInformationHelper.DIV_TARGET,
                          TopologicalInformationHelper.LOG_TARGET,
                          TopologicalInformationHelper.LOG_DIV_TARGET,
                          TopologicalInformationHelper.LOG_DIV_WD]:
            transform_name, ylabel = TopologicalInformationHelper.get_transform_name(transform)
            for hd in range(max_dim):
                all_curves = []
                all_labels = []
                # all_labels.append(pe_labels[hd])
                all_labels.append(ps_labels[hd])
                all_labels.append(fk_labels[hd])
                all_labels.append(wd_labels[hd])
                all_labels.append(hwd_labels[hd])
                all_labels.append(swd_labels[hd])
                for i,M in enumerate(slices):
                    all_labels.append(swdc_labels[i * max_dim + hd])

                all_labels.append(etd_labels[hd])

                for angle in range(asize):
                    all_labels.append(etda_labels[angle * max_dim + hd])
                for layer in range(nlayers):
                    xpoints = []
                    # xpoints.append(pe_curves[layer][hd])
                    xpoints.append(TopologicalInformationHelper.apply_transform(wd_curves[layer][hd],
                                                                                ps_curves[layer][hd],
                                                                                transform))
                    xpoints.append(TopologicalInformationHelper.apply_transform(wd_curves[layer][hd],
                                                                                fk_curves[layer][hd],
                                                                                transform))
                    xpoints.append(TopologicalInformationHelper.apply_transform(wd_curves[layer][hd],
                                                                                wd_curves[layer][hd],
                                                                                transform))
                    xpoints.append(TopologicalInformationHelper.apply_transform(wd_curves[layer][hd],
                                                                                hwd_curves[layer][hd],
                                                                                transform))
                    xpoints.append(TopologicalInformationHelper.apply_transform(wd_curves[layer][hd],
                                                                                swd_curves[layer][hd],
                                                                                transform))

                    for i,M in enumerate(slices):
                        xpoints.append(TopologicalInformationHelper.apply_transform(wd_curves[layer][hd],
                            swd_collection[layer][i*max_dim + hd], transform)
                        )

                    xpoints.append(TopologicalInformationHelper.apply_transform(wd_curves[layer][hd],
                                                                                etd_curves[layer][hd],
                                                                                transform))

                    for angle in range(asize):
                        xpoints.append(TopologicalInformationHelper.apply_transform(wd_curves[layer][hd],
                            etda_curves[layer][angle*max_dim + hd], transform)
                        )

                    all_curves.append(xpoints)

                InformationVisualizationHelper.plot_persistence_curves(all_curves, all_labels,
                                                                       f"Topological curves in H_{hd}",
                                                                       ylabel,
                                                                       activation_function,
                                                                       result_folder,
                                                                       approach,
                                                                       transform_name)

        for key in TopologicalInformationHelper.time_in_seconds:
            file.write(f"{key}: {TopologicalInformationHelper.time_in_seconds[key]} milliseconds\n")
        file.close()

if __name__ == '__main__':
    from ae_handler import *
    from utilities.directoy_helper import DirectoryHelper

    overall_path = "results/"
    folders = DirectoryHelper.get_all_subfolders(root_path=overall_path, dir_pattern="AE_", ignore_pattern = "Autoencoder")
    policy = 0
    for folder_name in folders:
        topological_path = f"{folder_name}/topological_info"
        td_path = "results/AutoencoderWeightTopologyApp/topological_distance_matrices"
        all_files = DirectoryHelper.get_all_filenames(root_path=topological_path, file_pattern=f"test")
        all_files = DirectoryHelper.sort_filenames_by_suffix(all_files, sep="_")
        act_function = folder_name.split("_")[1].upper()

        if act_function == "RELU":
            continue

        # get all pd
        all_pds = TDAHelper.get_all_pdiagrams(all_files, output_filename=None)

        angle_set = ExtendedTopologyDistanceHelper.compute_angle_dict()
        approach = 2 - policy

        TopologicalInformationHelper.draw_all_topo_curves_splitted_by_hd(all_pds, activation_function=act_function,
                                                                         result_folder=td_path,
                                                                         angle_set=angle_set,
                                                                         approach=approach)
        print(f"processed {act_function}")

from experiments.base_experiments import BaseExperiment
# -*- coding: utf-8 -*-
import os

import numpy as np

from utilities.directoy_helper import DirectoryHelper
from utilities.tda_helper import TDAHelper
from utilities.topological_information_provider import TopologicalInformationHelper
from utilities.ae_handler import AETrainerController, ActivationFunctionTypes
from utilities.extended_topology_distance import ExtendedTopologyDistanceHelper

class AEExperiment(BaseExperiment):
    '''
    This experiment assumes that Persistence Diagrams per layer per activation function were computed
    and are stored in datasets/autoencoders.

    This enum is to define the comparison policy
    All versus first: means that the persistence diagram of each layer will be
    compared against the PD of the input layer.

    The other approach means that we compare each PD with the previous one, the first one is compared with it self
    '''
    VERSUS_FIRST, VERSUS_PREVIOUS = range(2)
    def __init__(self, ae_controller: AETrainerController, p, policy=None):
        self.policy = AEExperiment.VERSUS_FIRST if policy is None else policy
        self.ae_controller = ae_controller
        super(AEExperiment, self).__init__(ae_controller.get_result_folder(), p)

    def policy_name(self):
        if self.policy == AEExperiment.VERSUS_FIRST:
            return "ALLVersusFirst"
        return "ALLVersusPrevious"

    def configure_experiment_name(self):
        return f"{str(self.__class__.__name__)}{self.policy_name()}"

    def execute(self):
        """
        Entry point. This method perform the whole process. This method assumes there is a data folder
        with the point clouds per layer or the persistence information per layer. The filename format was specified
        on the load_files function.

        The first step is data loading. We can load point cloud info and also persistent homology information
        per activation function.

        Then we can visualize co-cycles or any supported persistence curve.

        :return:
        """
        aft_folders = self.ae_controller.get_generated_folders()

        policy = self.policy
        for aft in aft_folders:
            folder_name = aft_folders[aft]
            self.topological_path = f"{folder_name}/topological_info"
            self.validate_topological_path() # verify if the corresponding diagrams were computed
            td_path = f"{folder_name}/{self.experiment_name}/{self.distance_folder}"
            all_files = DirectoryHelper.get_all_filenames(root_path=self.topological_path, file_pattern=f"test")
            all_files = DirectoryHelper.sort_filenames_by_suffix(all_files, sep="_")

            unpref_name = ActivationFunctionTypes.get_unprefixed_name(aft)

            # get all pd
            all_pds = TDAHelper.get_all_pdiagrams(all_files, output_filename=None)

            angle_set = ExtendedTopologyDistanceHelper.compute_angle_dict()
            approach = 2 - policy # 2 = ALL_VERSUS_FIRST, 1 = ALL_VERSUS_PREVIOUS

            TopologicalInformationHelper.draw_all_topo_curves_splitted_by_hd(all_pds, activation_function=unpref_name,
                                                                             result_folder=td_path,
                                                                             angle_set=angle_set,
                                                                             approach=approach)
            print(f"processed {unpref_name}")





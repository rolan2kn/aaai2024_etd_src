import os.path
import time

import numpy as np

from utilities.directoy_helper import DirectoryHelper
from utilities.extended_topology_distance import ExtendedTopologyDistanceHelper
from utilities.tda_helper import TDAHelper
from utilities.distance_matrix_helper import DistanceMatrixHelper


class BaseExperiment:
    def __init__(self, result_folder, p=2,
                 save_diags=False):
        self.experiment_name = self.configure_experiment_name()
        self.topological_folder = "topological_info"
        self.topological_path = None
        self.result_folder = result_folder
        self.save_diags = save_diags
        self.p = p

        self.distance_folder = "pd_distance_matrices/"
        self.td_path = f"{self.result_folder}"

    def configure_experiment_name(self):
        return str(self.__class__.__name__)

    def validate_topological_path(self):
        if self.topological_path is None or not os.path.isdir(self.topological_path):
            raise Exception("You must compute your persistence diagrams before ask for experiments")

        return

    def execute(self):
        pass



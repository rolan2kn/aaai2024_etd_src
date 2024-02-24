
import multiprocessing
import os

import gudhi
import numpy as np
from gudhi.representations import PersistenceImage, Entropy, Landscape
from sklearn.metrics import pairwise_distances

from utilities.directoy_helper import DirectoryHelper
from utilities.distance_matrix_helper import DistanceMatrixHelper
from utilities.persistence_diagram_helper import PersistenceDiagramHelper
from utilities.information_visualization_helpers import InformationVisualizationHelper


class TDAHelper:
    CUBICAL, RIPS, SPARSE_RIPS, ALPHA, WITNESS = range(5)
    def __init__(self, **kwargs):
        '''
        '''
        self.diags = []
        self.input_data = kwargs.get("input_data", None)
        self.input_data_name = kwargs.get("input_data_filename", None)
        self.force_diags = kwargs.get("force_diags", False)
        if self.input_data_name is not None:
            self.input_data_name = self.input_data_name.replace("//", "/")

        self.max_dim = kwargs.get("max_dim", 2)
        self.diags_filename = None
        self.sparse_value = kwargs.get("sparse_value", 0.5)
        self.draw_diagrams = kwargs.get("draw_diagrams", True)
        self.enable_collapse = kwargs.get("enable_collapse", None)
        self.max_value = kwargs.get("max_value", None)
        self.metric = kwargs.get("metric", "manhattan")
        self.enable_max_edge = self.max_value is not None
        self.output_folder = kwargs.get("output_folder", None)
        self.compute_hks = kwargs.get("compute_hks", True)

    @staticmethod
    def sanity_intervals(PD):
        '''
        Remove infinite death times
        :param PD:
        :return:
        '''

        return PersistenceDiagramHelper.truncate_diagram(PD)

    def load_input_data(self):
        '''
        This method load the corresponding input data using its name
        :return:
        '''
        if self.input_data is not None:
            return self.input_data

        return DistanceMatrixHelper.load_data_collection(self.input_data_name)

    def build_pd_filename(self):
        '''
        We use this function to construct a unique name for the corresponding persistence diagram

        :return:
        '''
        dm_name = 'pd'
        parts = []
        if self.input_data_name is not None:
            parts = self.input_data_name.split("/")
            dm_name = parts[-1]
            if dm_name == '':
                dm_name = parts[-2]
        if self.output_folder is None:
            pos = 3 if len(parts) > 0 and len(parts[-1]) > 0 else 4
            self.output_folder = f"{'/'.join(parts[:0-pos])}/topological_info"

        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)

        self.diags_filename = f"{self.output_folder}/{dm_name}.diag.pickle"

        return self.diags_filename

    def were_pd_already_computed(self):
        '''
        This method determines if the persistence diagram was already computed
        :return:
        '''
        current_name = self.build_pd_filename()

        if not self.force_diags and os.path.exists(current_name):
            return True

        return False

    def compute_persistence(self):
        '''
        We compute the persistence diagram in case it is not exists

        When the PD is computed we also save useful information such as:
        Persistence diagram, Barcode, and the diagrams information
        :return:
        '''
        if self.were_pd_already_computed():
            return TDAHelper.load_diags(self.diags_filename)

        D = self.load_input_data()
        if D is None:
            raise Exception("We were expected a valid distance matrix!")

        complex = gudhi.RipsComplex(distance_matrix=D)
        if self.enable_collapse is None:
            self.simplex_tree = complex.create_simplex_tree(max_dimension=self.max_dim)
        else:
            self.simplex_tree = complex.create_simplex_tree(max_dimension=float(1))
            self.simplex_tree.collapse_edges()#self.enable_collapse)
            self.simplex_tree.expansion(self.max_dim)

        diags = self.simplex_tree.persistence()

        # truncate and grouped intervals by dimensions
        self.diags = self.sanity_intervals(diags)
        if self.draw_diagrams:
            InformationVisualizationHelper.save_diagrams_as_images(self.diags_filename, diags)
        del diags
        TDAHelper.save_diags(self.diags, self.diags_filename)

        return self.diags

    def compute_alpha_off_persistence(self):
        """
        We build the Alpha complex directly from the off file,
        considering the off-file points
        """
        if self.were_pd_already_computed():
            return TDAHelper.load_diags(self.diags_filename)

        D = self.load_input_data()  # we expect an off filename
        if D is None or D.find(".off") == -1:
            raise Exception("We were expected a valid distance matrix!")

        points = gudhi.read_points_from_off_file(off_file=D)
        self.simplex_tree = gudhi.AlphaComplex(points = points).create_simplex_tree()
        diags = self.simplex_tree.persistence()

        # truncate and grouped intervals by dimensions
        self.diags = self.sanity_intervals(diags)
        if self.draw_diagrams:
            InformationVisualizationHelper.save_diagrams_as_images(self.diags_filename, diags)
        del diags
        TDAHelper.save_diags(self.diags, self.diags_filename)

        return self.diags

    def merge_with_heat_kernel_signature(self, filename, points):
        import trimesh
        from utilities.mesh_signature_helper import SignatureExtractor

        if not self.compute_hks:
            return points

        mesh = trimesh.load(filename)

        extractor = SignatureExtractor(mesh, 100, "beltrami")

        # we generate a heat kernel signature with 3 dimensions
        # and we used it as a suffix of each sample point leading to R6 point cloud
        hks = extractor.heat_signatures(3)

        r6_points = []
        no_points = len(points)
        for j in range(no_points):
            p = list(points[j])
            p.extend(hks[j])
            r6_points.append(p)

        return r6_points

    def compute_sparse_rips_persistence(self):
        '''
        We compute the persistence diagram in case it is not exists

        When the PD is computed we also save useful information such as:
        Persistence diagram, Barcode, and the diagrams information
        :return:
        '''
        if self.were_pd_already_computed():
            return TDAHelper.load_diags(self.diags_filename)

        D = self.load_input_data()
        if D is None:
            raise Exception("We were expected a valid distance matrix!")
        if type(D) == str and D.find(".off") != -1:
            points = gudhi.read_points_from_off_file(off_file=D)

            points = self.merge_with_heat_kernel_signature(filename=D, points=points)
            del D

            D = pairwise_distances(points, n_jobs=-1, metric=self.metric)
            self.max_value = np.quantile(D, 0.75)  # we use the 0.75 quantile as max value
            self.enable_max_edge = self.max_value is not None
        if self.enable_max_edge:
            '''
            See Gudhi Rips complex python documentation https://gudhi.inria.fr/python/latest/rips_complex_user.html
            for a detailed exposition.
            According to Gudhi sparse value should not surpass 1, so we use 0.5
            '''
            complex = gudhi.RipsComplex(distance_matrix=D, max_edge_length=self.max_value, sparse=self.sparse_value)
        else:
            complex = gudhi.RipsComplex(distance_matrix=D, sparse=self.sparse_value)
        if self.enable_collapse is None:
            self.simplex_tree = complex.create_simplex_tree(max_dimension=self.max_dim)
        else:
            self.simplex_tree = complex.create_simplex_tree(max_dimension=float(1))
            self.simplex_tree.collapse_edges()  # self.enable_collapse)
            self.simplex_tree.expansion(self.max_dim)

        diags = self.simplex_tree.persistence()

        # truncate and grouped intervals by dimensions
        self.diags = self.sanity_intervals(diags)
        if self.draw_diagrams:
            InformationVisualizationHelper.save_diagrams_as_images(self.diags_filename, diags)
        del diags
        TDAHelper.save_diags(self.diags, self.diags_filename)

        return self.diags

    def compute_diagrams_by_type(self, complex_type):
        if complex_type == TDAHelper.RIPS:
            return self.compute_persistence()
        if complex_type == TDAHelper.ALPHA:
            return  self.compute_alpha_off_persistence()
        if complex_type == TDAHelper.SPARSE_RIPS:
            return self.compute_sparse_rips_persistence()
        if complex_type == TDAHelper.CUBICAL:
            return self.compute_cubical_persistence()

        return []

    def compute_cubical_persistence(self):
        '''
        We compute the persistence diagram in case it is not exists

        When the PD is computed we also save useful information such as:
        Persistence diagram, Barcode, and the diagrams information
        :return:
        '''
        if self.were_pd_already_computed():
            return TDAHelper.load_diags(self.diags_filename)

        image = self.load_input_data()
        extra_dim = []
        if len(image.shape) > 1:
            image = image.flatten()
        X, = image.shape
        image = [image]

        nrow = int(np.ceil(np.sqrt(X)))
        dimensions = [nrow, nrow]
        dimensions.extend(extra_dim)

        data = []
        for frame in image:
            values = list(frame)
            values.extend(np.zeros(nrow ** 2 - X))
            data.extend(values)

        cubical_complex = gudhi.CubicalComplex(dimensions=dimensions,
                                               top_dimensional_cells=data)

        diags = cubical_complex.persistence()

        self.diags = self.sanity_intervals(diags)
        if self.draw_diagrams:
            InformationVisualizationHelper.save_diagrams_as_images(self.diags_filename, diags)
        del diags
        TDAHelper.save_diags(self.diags, self.diags_filename)

        return self.diags

    @staticmethod
    def save_diags(diags, filename):
        '''
        We save the current persistence diagram to filename path.

        :return:
        '''
        return DistanceMatrixHelper.save_data_collection(diags,
                                                  filename)

    @staticmethod
    def load_diags(filename):
        '''
        We load a persistence diagram from its corresponding filename

        :return:
        '''
        if filename is None or not os.path.isfile(filename):
            raise Exception(f"{filename} not found!")

        if filename.find(".pickle") != -1:
            # if it was stored with pickle
            return DistanceMatrixHelper.load_data_collection(filename)

        return None

    @staticmethod
    def get_all_pdiagrams(pd_all_diagnames, output_filename = None):
        if output_filename is not None and (os.path.isfile(output_filename) or os.path.isfile(f"{output_filename}.pickle")):
            return DistanceMatrixHelper.load_data_collection(output_filename)

        all_pds = []

        with multiprocessing.Pool() as pool:
            for diag in pool.map(TDAHelper.load_diags, pd_all_diagnames):
                all_pds.append(diag)

        if output_filename is not None and len(all_pds) > 0:
            return DistanceMatrixHelper.save_data_collection(all_pds, output_filename)

        return all_pds

    @staticmethod
    def compute_one_pd(items):
        data, result_folder, complex_type, filename, max_dim = items
        tda_helper = TDAHelper(input_data_filename=filename,
                               input_data=data,
                               max_dim=max_dim, draw_diagrams=False,
                               output_folder=f"{result_folder}/topological_info/")
        tda_helper.compute_diagrams_by_type(complex_type)

        return tda_helper.diags_filename

    @staticmethod
    def compute_all_diagrams(data_processor, max_dim):
        result_folder = data_processor.get_result_folder()

        all_pds = DirectoryHelper.get_all_filenames(f"{result_folder}/topological_info/", file_pattern=".diag")
        if len(all_pds) > 0:
            return []

        train_imgs, test_imgs = data_processor.execute()
        complex_type = data_processor.get_complex_type()

        all_pds = []
        arg_items = [(image, result_folder, complex_type, f"pd_train_{id}", max_dim)
                        for id, image in enumerate(train_imgs)]

        with multiprocessing.Pool() as pool:
            for diag_name in pool.map(TDAHelper.compute_one_pd, arg_items):
                #del diag
                print(diag_name)
                all_pds.append(diag_name)

        del arg_items
        arg_items = [(image, result_folder, complex_type, f"pd_test_{id}", max_dim)
                     for id, image in enumerate(test_imgs)]

        with multiprocessing.Pool() as pool:
            for diag_name in pool.starmap(TDAHelper.compute_one_pd, arg_items):
                #del diag
                print(diag_name)
                all_pds.append(diag_name)

        return all_pds

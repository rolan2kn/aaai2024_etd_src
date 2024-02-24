
import os
import sys
import numpy as np


class DirectoryHelper:
    @staticmethod
    def create_dir(folder):
        os.makedirs(folder)

    @staticmethod
    def get_home_dir():
        """Return the user home directory"""
        return os.path.expanduser('~')

    @staticmethod
    def get_app_home(app_name):
        """Return the application home directory. This will be a directory
        in $HOME/.app_name/ but with app_name lower cased.
        """
        return os.path.join(DirectoryHelper.get_home_dir(), '.' + app_name.lower())

    @staticmethod
    def get_pycharm_path():
        """
        compute the path of all python projects PATHS/projects/pycharm
        :return:
        """
        m_path = DirectoryHelper.get_module_path()
        pycharm_path = m_path

        if len(m_path) > 0:
            pycharm_path = "/".join(m_path.split("/")[:-1])

        return pycharm_path

    @staticmethod
    def get_pycharm_project_path(pchrm_project):
        """
        compute the path of a pycharm project
        :return:
        """

        return "{0}/{1}".format(DirectoryHelper.get_pycharm_path(), pchrm_project)

    @staticmethod
    def get_remote_module_handler(project_name, filename, module_name):
        import importlib.machinery

        sphere_path = "{0}/{1}".format(DirectoryHelper.get_pycharm_project_path(project_name),
                                       filename)
        loader = importlib.machinery.SourceFileLoader(module_name, sphere_path)

        return loader.load_module(module_name)

    @staticmethod
    def get_root_dir():
        """Return the root directory of the application"""

        # was run from an executable?

        root_d = None#os.path.dirname(sys.argv[0])

        if not root_d:
            try:
                path = os.path.abspath(__file__)
                name = __name__.replace(".", "/")
                pos = path.find(name)
                if pos != -1:
                    root_d = path[:pos]
            except:
                path = os.path.dirname(__file__)
                root_d = '/'.join(path.split('/')[:-1])

        return root_d

    @staticmethod
    def get_module_path():
        '''Get the path to the current module no matter how it's run.'''

        # if '__file__' in globals():
        #     If run from py
        #    # return os.path.dirname(__file__)

        # If run from command line or an executable

        return DirectoryHelper.get_root_dir()

    @staticmethod
    def get_module_pkg():
        """Return the module's package path.
        Ej:
            if current module is the the call to:
                get_module_pkg()
            should return:
        """

        return '.'.join(__name__.split('.')[:-1])

    @staticmethod
    def separate_filename_and_path(full_filename):
        if full_filename.find("/") == -1:
            return "", full_filename

        parts = full_filename.split("/")
        full_path = "/".join(parts[:-1])
        filename = parts[-1]

        return full_path, filename

    @staticmethod
    def get_all_filenames(root_path=None, file_pattern = None, ignore_pattern = None):
        root_path = DirectoryHelper.get_module_path() if root_path is None else root_path
        files_list = []

        for path, _, files in os.walk(root_path):
            for _file in files:
                str_to_print = "{0}/{1}".format(path, _file)
                if ignore_pattern is None or str_to_print.find(ignore_pattern) == -1:
                    if file_pattern is None or len(file_pattern) == 0:
                        files_list.append(str_to_print)
                    elif str_to_print.find(file_pattern) != -1:
                        files_list.append(str_to_print)

        return files_list

    @staticmethod
    def get_all_subfolders(root_path=None, dir_pattern=None, ignore_pattern=None):
        root_path = DirectoryHelper.get_module_path() if root_path is None else root_path
        dir_list = []
        for file in os.listdir(root_path):
            d = os.path.join(root_path, file)
            if not os.path.isdir(d):
                continue
            if ignore_pattern is not None and d.find(ignore_pattern) != -1:
                continue
            if dir_pattern is None or len(dir_pattern) == 0 or d.find(dir_pattern) != -1:
                dir_list.append(os.path.normpath(d))

        return dir_list

    @staticmethod
    def remove_extension(filename):
        parts = filename.split(".")
        return '.'.join(parts[:-1])

    @staticmethod
    def sort_filenames_by_suffix(all_filenames, sep):
        idxs = list(range(len(all_filenames)))
        paths = [
                DirectoryHelper.extract_value_from_path(filename, sep)
                for filename in all_filenames
                ]

        sorted_by_Xs = sorted(
            list(zip(paths, idxs)))

        sorted_paths = []
        for _, id in sorted_by_Xs:
            sorted_paths.append(all_filenames[id])

        del all_filenames
        del paths
        del idxs
        del sorted_by_Xs

        return sorted_paths

    @staticmethod
    def extract_value_from_path(desired_path, sep):
        if desired_path is not None and len(desired_path) > 0:
            pos = desired_path.rfind(sep)
            if pos != -1:  # a valid folder
                desired_path = desired_path[pos + len(sep):]
                pos2 = desired_path.find('.')
                if pos2 != -1:
                    desired_path = desired_path[:pos2]
                if desired_path.isdigit():
                    return int(desired_path)

                # we assume that the desired_path prefix is always a number,
                # so we only need to extract it
                size = len(desired_path)
                digit_part = ''
                for i in range(size):
                    if desired_path[i] >= '0' and desired_path[i] <= '9':
                        digit_part += desired_path[i]
                    else:
                        break
                if len(digit_part) > 0:
                    return int(digit_part)

        return None  # otherwise

    @staticmethod
    def load_pd_files(folder=None, pattern=None, get_items=True, dim_reduction_type=None):
        '''
        This method creates a dictionary with the information contained in desired files on a specific folder.
        The filenames must have the following format:

         <LAYER POSITION>.<LAYER NAME>-<INFORMATION TYPE>.<EXTENSION>

         LAYER POSITION: Specify the layer's position because it is impossible to unravel the appropriate ordering
                         of the layer information from the filename or the contained data inside the file. Since we
                          evaluate the evolution and changes of the topology from layer to layer, it is mandatory
                          to order layers properly. In another case, the resulting plots could not have a reliable
                           interpretation.

        LAYER NAME: It will be used to identify the layer specific information.
        INFORMATION TYPE: Is a suffix that represents the information type: 'spheres' represents the data points,
                          and 'ripser' represents the persistence information associated.

        EXTENSION: file extension, we support *.npy and *.pkl files

        Examples:

        1. inputs-spheres.npy
        1. inputs-spheres-ripser.npy

        2. enc1-spheres.npy
        2. enc1-spheres-ripser.npy


        :param folder: files location
        :param pattern: sub-string to identify desired files
        :param get_items: flag to determine the kind of argument that we need. Usually is applied to separate picke
                         items from numpy files .npy
        :param dim_reduction_type: enumerate which indicates the dimensonality reduction method to apply TSNE, UMAP,
        PCA.
                                   If no dimensionality reduction is required this param should be None.
        :return: The dictionary with all data per layer, the filename is the layer.
        '''
        data = {}
        if pattern.find("pkl") == -1:
            desired_files = DirectoryHelper.get_all_filenames(root_path=folder, file_pattern=pattern)

            for filename in desired_files:
                if get_items:
                    xx = np.load(filename, allow_pickle='TRUE').item()

                else:
                    xx = np.load(filename)

                data.update({filename: xx})
        else:
            import pickle
            diagrams = pickle.load(open(f"{folder}/diagrams_lrelu.pkl", "rb"))

            desired_filenames = DirectoryHelper.get_all_filenames(root_path=folder, file_pattern="ripser.npy")
            desired_filenames = sorted(desired_filenames)
            for ii, layerd in enumerate(diagrams):
                key_name = desired_filenames[ii]
                data.update({key_name: {"dgms": np.array(layerd, dtype=object)}})

        return data

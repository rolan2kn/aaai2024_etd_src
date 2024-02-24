
import os.path
import cv2 as cv
import os
import random
from collections import Counter

import numpy as np
from tqdm import tqdm

from utilities.directoy_helper import DirectoryHelper
from utilities.outer_reader import OutexReaderHelper
from utilities.torch_datasets_reader import FashionMNist, Cifar10, ImageDataPreprocessor


class DatasetProcessor:
    def __init__(self, overall_path):
        self.result_folder = self.resolve_result_path(overall_path)

        if not os.path.isdir(self.result_folder):
            os.makedirs(self.result_folder)

        self.train_indexes = []
        self.test_indexes = []
        self.y_train = []
        self.y_test = []
        self.train = []
        self.test = []

    def resolve_result_path(self, overall_path):
        return f"{overall_path}/{self.get_name()}"

    def get_y_train(self):
        raise Exception("You must implement this method in a derived class")

    def get_y_test(self):
        raise Exception("You must implement this method in a derived class")

    def get_chosen_train_labels(self):
        y_train = np.array(self.get_y_train())

        return y_train[self.train_indexes]

    def get_chosen_test_labels(self):
        y_test = np.array(self.get_y_test())

        return y_test[self.test_indexes]

    def get_complex_type(self):
        pass

    def get_classes(self):
        return np.unique(self.y_train)
    def set_train_and_test_sets(self, new_train, new_test):
        self.train = new_train
        self.test = new_test

    def get_chosen_train_samples(self):
        train = np.array(self.train)

        return train[self.train_indexes]

    def get_chosen_test_samples(self):
        test = np.array(self.test)

        return test[self.test_indexes]

    def get_result_folder(self):
        return self.result_folder

    def execute(self):
        return [],[]

    def generate_subsampling_indexes(self, no_train_samples, no_test_samples):
        '''
        This method performs a subsampling of a dataset in the following way:

        1. for labeled data we choose a collection of len_train samples per class.
        2. for unlabeled data we extract len_test samples.

        Since labels and samples are aligned by knowing the desired indexes we can
        extract labels and samples.

        This method assumes that we already have labeling information.
        '''
        y_train = self.get_y_train()
        y_test = self.get_y_test()

        self.train_indexes = []
        self.test_indexes = []

        if no_train_samples is not None and self.has_a_train_set():
            class_separation = {int(id): 0 for id in np.unique(y_train)}

            # Separate all pdiagrams according the class it belongs
            for id, ci in enumerate(y_train):
                ci = int(ci)
                if class_separation[ci] < no_train_samples:
                    class_separation[ci] += 1
                    self.train_indexes.append(int(id))
        else:
            self.train_indexes = list(range(len(y_train)))
        if no_test_samples is not None and self.has_a_test_set():
            self.test_indexes = []
            rng = np.random.default_rng(seed=123)
            self.test_indexes = rng.choice(len(y_test), no_test_samples)
        else:
            self.test_indexes = list(range(len(y_train)))
        return self.train_indexes, self.test_indexes

    def has_a_test_set(self):
        if self.y_test is None:
            return False
        return len(self.y_test)

    def has_a_train_set(self):
        if self.y_train is None:
            return False

        return len(self.y_train)

    def get_name(self):
        return str(self.__class__.__name__)


class Shrec07Processor(DatasetProcessor):
    def __init__(self, overall_path):
        super(Shrec07Processor, self).__init__(overall_path)

    def execute(self):
        if len(self.train) > 0:
            return self.train, self.test

        module_path = DirectoryHelper.get_module_path()
        offpath = f"{module_path}/datasets/Shrec07/off/"
        # offpath = module_path

        tmp_off_files = DirectoryHelper.get_all_filenames(offpath, file_pattern=".off")
        all_off_files = DirectoryHelper.sort_filenames_by_suffix(tmp_off_files, sep="/")

        self.train = all_off_files
        self.test = []

        return all_off_files,[]

    def generate_subsampling_indexes(self, no_train_samples, no_test_samples):

        y_train = self.get_y_train()
        self.get_y_test()

        self.train_indexes = []
        self.test_indexes = []

        # test elements are taken from the complement of the train set
        full_test_indexes = []

        if no_train_samples is not None and self.has_a_train_set():
            class_separation = {id: 0 for id in np.unique(y_train)}

            # Separate all pdiagrams according the class it belongs
            rng = np.random.default_rng(seed=42)
            label_size = len(self.labels)
            tmp_train = []
            tmp_test = []
            seed_set = range(self.samples_per_class)

            for c in range(label_size):
                permutation = rng.permutation(seed_set)
                c_train_choice = permutation[:no_train_samples]
                c_test_choice = permutation[no_train_samples:]

                tmp_train.append(c_train_choice+(c*self.samples_per_class))
                tmp_test.append(c_test_choice+(c*self.samples_per_class))

            self.train_indexes = np.concatenate(tmp_train)
            full_test_indexes = np.concatenate(tmp_test)

            del tmp_train
            del tmp_test

            # if no desired number of test set we take the full index set by default
            self.test_indexes = full_test_indexes
            if no_test_samples is not None:
                # if there s a desired number of train set we choose it at random
                random.seed(42)
                self.test_indexes = random.sample(list(full_test_indexes), no_test_samples)

        return self.train_indexes, self.test_indexes

    def get_y_train(self):
        '''
        1 -  20  Human
        21 -  40  Cup
        41 -  60  Glasses
        61 -  80  Airplane
        81 - 100  Ant
        101 - 120  Chair
        121 - 140  Octopus
        141 - 160  Table
        161 - 180  Teddy
        181 - 200  Hand
        201 - 220  Plier
        221 - 240  Fish
        241 - 260  Bird
       (261 - 280) Spring (excluded from our study)
        281 - 300  Armadillo
        301 - 320  Bust
        321 - 340  Mech
        341 - 360  Bearing
        361 - 380  Vase
        381 - 400  Fourleg
        '''

        if self.has_a_train_set():
            return self.y_train

        self.labels = ["Human",
                    "Cup",
                    "Glasses",
                    "Airplane",
                    "Ant",
                    "Chair",
                    "Octopus",
                    "Table",
                    "Teddy",
                    "Hand",
                    "Plier",
                    "Fish",
                    "Bird",
                    "Armadillo",
                    "Bust",
                    "Mech",
                    "Bearing",
                    "Vase",
                    "Fourleg"]
        self.samples_per_class = 20
        self.y_train = []
        for i in range(len(self.labels)):
            self.y_train.extend([i]*self.samples_per_class)
        return self.y_train

    def get_y_test(self):
        if self.has_a_test_set():
            return self.y_test

        # we assign the same labels than train
        # in order to choose from here using the index set
        self.y_test = self.get_y_train()

        return self.y_test

    def get_complex_type(self):
        '''
        This indicates which kind of complex we should build
        '''
        from utilities.tda_helper import TDAHelper

        return TDAHelper.SPARSE_RIPS


class OutexProcessor(DatasetProcessor):
    def __init__(self, overall_path):
        super(OutexProcessor, self).__init__(overall_path)
        module_path = DirectoryHelper.get_module_path()
        self.outex_reader = OutexReaderHelper(module_path)

    def execute(self):
        if len(self.train) == 0:
            self.train, self.test = self.outex_reader.load_data()

        return self.train, self.test

    def get_y_train(self):
        if not self.has_a_train_set():
            self.y_train = self.outex_reader.load_train_labels()
            self.labels = np.unique(self.y_train)
            counter = Counter(self.y_train)
            self.samples_per_class = counter[0] # based on Outex is balanced
        return self.y_train

    def get_y_test(self):
        if not self.has_a_test_set():
            self.y_test = self.outex_reader.load_test_labels()

        return self.y_test

    def get_complex_type(self):
        '''
        This indicates the family of complex we should build
        '''
        from utilities.tda_helper import TDAHelper

        return TDAHelper.CUBICAL

    def generate_subsampling_indexes(self, no_train_samples, no_test_samples):

        y_train = self.get_y_train()
        y_test = self.get_y_test()

        self.train_indexes = []
        self.test_indexes = []
        desired_label = 30

        # test elements are taken from the complement of the train set
        if no_train_samples is not None and self.has_a_train_set():
            # Separate all pdiagrams according the class it belongs
            rng = np.random.default_rng(seed=42)
            label_size = len(self.labels[:desired_label])
            tmp_train = []

            for c in range(label_size):
                c_train_choice = rng.choice(self.samples_per_class, no_train_samples, replace=False)
                tmp_train.append(c_train_choice+(c*self.samples_per_class))

            self.train_indexes = np.concatenate(tmp_train)
            del tmp_train

            # if no desired number of test set we take the full index set by default
            tmp_test = []
            for id, c in enumerate(self.y_test):
                if c < desired_label:
                    tmp_test.append(id)
            self.test_indexes = range(len(y_test))
            if no_test_samples is not None:
                # if there is a desired number of train set we choose it at random
                self.test_indexes = rng.choice(range(len(tmp_test)), no_test_samples, replace=False)
        else:
            self.train_indexes = list(range(len(self.y_train)))
            self.test_indexes = list(range(len(self.y_test)))

        return self.train_indexes, self.test_indexes

class FashionMNistProcessor(DatasetProcessor):
    def __init__(self, overall_path, transform_type=None):

        self.transform_type = transform_type
        super(FashionMNistProcessor, self).__init__(overall_path)

    def resolve_result_path(self, overall_path):
        type_name = ImageDataPreprocessor.typename(self.transform_type)
        return f"{overall_path}/{self.get_name()}/{type_name}"
    def execute(self):
        if len(self.train) > 0:
            return self.train, self.test

        fashion = FashionMNist(transform_type=self.transform_type)

        self.train, self.y_train = fashion.xtrain, fashion.ytrain
        self.test, self.y_test = fashion.xtest, fashion.ytest

        return self.train, self.test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    def get_complex_type(self):
        '''
        This indicates which kind of complex we should build
        '''
        from utilities.tda_helper import TDAHelper

        return TDAHelper.CUBICAL

class Cifar10Processor(DatasetProcessor):
    def __init__(self, overall_path, transform_type=None):
        self.transform_type = transform_type
        super(Cifar10Processor, self).__init__(overall_path)

    def resolve_result_path(self, overall_path):
        type_name = ImageDataPreprocessor.typename(self.transform_type)
        return f"{overall_path}/{self.get_name()}/{type_name}"
    def execute(self, **kwargs):
        if len(self.train) > 0:
            return self.train, self.test

        cifar10 = Cifar10(transform_type=self.transform_type)

        self.train, self.y_train = cifar10.xtrain, cifar10.ytrain
        self.test, self.y_test = cifar10.xtest, cifar10.ytest

        return self.train, self.test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    def get_complex_type(self):
        '''
        This indicates which kind of complex we should build
        '''
        from utilities.tda_helper import TDAHelper

        return TDAHelper.CUBICAL

class DataProcessorFactory:
    (OUTEX, SHREC07, FASHION, CIFAR10,
     FASHION_HOG, CIFAR10_HOG, FASHION_IGG,
     CIFAR10_IGG, FASHION_GAUSS, CIFAR10_GAUSS) = range(10)

    @staticmethod
    def get_data_processor(overall_path, data_type):
        if data_type == DataProcessorFactory.OUTEX:
            return OutexProcessor(overall_path)
        if data_type == DataProcessorFactory.SHREC07:
            return Shrec07Processor(overall_path)
        if data_type == DataProcessorFactory.FASHION:
            return FashionMNistProcessor(overall_path, ImageDataPreprocessor.NONE)
        if data_type == DataProcessorFactory.CIFAR10:
            return Cifar10Processor(overall_path, ImageDataPreprocessor.NONE)
        if data_type == DataProcessorFactory.FASHION_IGG:
            return FashionMNistProcessor(overall_path, ImageDataPreprocessor.IGG)
        if data_type == DataProcessorFactory.CIFAR10_IGG:
            return Cifar10Processor(overall_path, ImageDataPreprocessor.IGG)
        if data_type == DataProcessorFactory.FASHION_GAUSS:
            return FashionMNistProcessor(overall_path, ImageDataPreprocessor.GAUSSIAN)
        if data_type == DataProcessorFactory.CIFAR10_GAUSS:
            return Cifar10Processor(overall_path, ImageDataPreprocessor.GAUSSIAN)
        if data_type == DataProcessorFactory.FASHION_HOG:
            return FashionMNistProcessor(overall_path, ImageDataPreprocessor.HOG)
        if data_type == DataProcessorFactory.CIFAR10_HOG:
            return Cifar10Processor(overall_path, ImageDataPreprocessor.HOG)

        return None

    @staticmethod
    def get_data_processor_settings(data_type):
        if data_type == DataProcessorFactory.OUTEX:
            return {"p":2, "no_train_samples":20, "no_test_samples": 200}

        if data_type == DataProcessorFactory.SHREC07:
            return {"p":2, "no_train_samples":15, "no_test_samples":10}
        if data_type == DataProcessorFactory.FASHION:
            return {"p":2, "no_train_samples":100, "no_test_samples":10}
        if data_type == DataProcessorFactory.FASHION_IGG:
            return {"p":2, "no_train_samples":100, "no_test_samples":10}
        if data_type == DataProcessorFactory.CIFAR10_IGG:
            return {"p":2, "no_train_samples":100, "no_test_samples":10}
        if data_type == DataProcessorFactory.FASHION_GAUSS:
            return {"p":2, "no_train_samples":100, "no_test_samples":10}
        if data_type == DataProcessorFactory.CIFAR10_GAUSS:
            return {"p":2, "no_train_samples":100, "no_test_samples":10}
        if data_type == DataProcessorFactory.FASHION_HOG:
            return {"p":2, "no_train_samples":100, "no_test_samples":10}
        if data_type == DataProcessorFactory.CIFAR10_HOG:
            return {"p":2, "no_train_samples":100, "no_test_samples":10}

        return {"p":2, "no_train_samples":None, "no_test_samples":None}


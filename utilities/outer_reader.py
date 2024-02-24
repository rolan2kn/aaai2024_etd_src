from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd


class OutexReaderHelper:
    def __init__(self, folder):
        self.folder = f"{folder}/datasets/outex_tc_00024/data"

    def load_image(self, img_name):
        path = f"{self.folder}/images"
        image = plt.imread(os.path.join(path, img_name))

        return image

    def load_data(self):
        path_data = f"{self.folder}/000/"

        train_img_names = np.array(pd.read_csv(path_data + "train.txt",
                                               sep=" ",
                                               usecols=[0]).to_numpy().flatten().tolist())
        test_img_names = np.array(pd.read_csv(path_data + "test.txt",
                                              sep=" ",
                                              usecols=[0]).to_numpy().flatten().tolist())

        train_img_names = train_img_names
        test_img_names = test_img_names

        train_im = []
        test_im = []
        for im in train_img_names:
            train_im.append(self.load_image(im))

        for im in test_img_names:
            test_im.append(self.load_image(im))

        return train_im, test_im

    def load_train_labels(self, size=None):
        path_data = f"{self.folder}/000/"

        all_labels = np.array(pd.read_csv(path_data + "train.txt",
                                          sep=" ",
                                          usecols=[1]).to_numpy().flatten().tolist())
        if size is not None:
            all_labels = all_labels[:size]

        return all_labels

    def load_test_labels(self, size=None):
        path_data = f"{self.folder}/000/"

        test_labels = np.array(pd.read_csv(path_data + "test.txt",
                                           sep=" ",
                                           usecols=[1]).to_numpy().flatten().tolist())

        if size is not None:
            return test_labels[:size]

        return test_labels



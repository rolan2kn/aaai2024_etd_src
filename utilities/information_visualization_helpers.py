# -*- coding: utf-8 -*-

import numpy as np
import itertools
from persim import plot_diagrams
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import gudhi
import seaborn as sns


class InformationVisualizationHelper:
    """
    The InformationVisualizationHelper class is intended for visualization purposes
    """

    @staticmethod
    def plot_persistence_curves(curves, labels, title, ylabel, activation_function,
                                result_folder = None, approach=2, transform = None):
        """
        Method for drawing persistence curves.

        :param curves: a collection of several curves to plot. By curve we meant a collection of values of a function f.
                       We can set f to be the Wasserstein distance W(di, dj), Persistent Entropy, or Topological
                       distance. Usually we have curves per homology group and also a
                       global curve which summarizes the whole topological information. In this setting, the curves
                       list will have d+k curves: d homology group curves, and k global curves. The curve format is
                       [[f0(x0), f0(x1), ..., f0(xN)], ..., [fM(x0), fM(x1), ..., fM(xN)]]
        :param labels: The relative name of each curve (homology group, global, etc).
        :param title: The name of the picture
        :param ylabel: The name of the curve function
        :param activation_function: current activation function name to differentiate the same plot between different
                                    activation functions.
        :return:
        """
        w, h = 16, 9
        if result_folder is None:
            result_folder = "./images/results/"

        lsize = 25
        area = 100

        fig = plt.figure(figsize=(w, h))
        ax = plt.subplot(111)
        size = len(curves)
        styles = ['solid', 'dashed', 'dashdot', 'dotted']
        marks = ['o', '*', 'v', '^', '-', '3', '8', "."]
        colors = ['red', 'green', 'blue', 'orange', 'purple',
                  'sandybrown', 'grey', 'violet', 'turquoise', 'blueviolet', 'darkgoldenrod' , 'bisque', 'olive', 'khaki']
        ss = len(styles)
        cc = len(colors)

        for i in range(1,
                       size):  # the idea is to plot a line connecting f(x-1), f(x), x coordinates are inferred by
            # positions
            item_list = curves[i]
            last_item = curves[i - 1]
            if type(item_list) not in (list, tuple, dict):
                item_list = [item_list]
                last_item = [last_item]
            hgroups = len(item_list)

            for j in range(0, hgroups):
                if i == 1:

                    ax.plot([i - 1, i], [last_item[j], item_list[j]], color=colors[j % cc] , linewidth=4,
                            linestyle=styles[j % ss], label=labels[j], marker=marks[j % ss], markersize=12)

                else:
                    ax.plot([i - 1, i], [last_item[j], item_list[j]], color=colors[j % cc], linewidth=4,
                            linestyle=styles[j % ss], marker=marks[j % ss], markersize=12)


        ax.set_xlabel("Layers", fontsize=lsize)
        ax.set_ylabel(ylabel, fontsize=lsize)
        k = 3 if len(curves) > 4 else 1

        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, fontsize=16,
                        title="Topological Curves",
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.1), ncol=k)
        text = ax.text(-0.2, 1.05, "", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid('on')
        name = title.replace(" ", "_").lower()
        approach_name = "all_vs_first" if approach == 2 else "each_vs_prev"
        transform = transform if transform is not None else ""
        fig_name = f"{result_folder}/{name}_curves_{activation_function}_{approach_name}_{transform}.png"
        fig.savefig(fig_name, bbox_extra_artists=(lgd, text), bbox_inches='tight')
        plt.close("all")

    @staticmethod
    def draw_point_clouds(full_pointcloud_data, result_folder = None):
        w, h = 16, 9
        fig = plt.figure(figsize=(w, h))
        nSpheres = len(full_pointcloud_data)
        alphas = [1, 0.3]

        if result_folder is None:
            result_folder = "./images/results/"

        point_labels = [(name.split("/")[-1])[:-3] for name in sorted(full_pointcloud_data)]
        for ii, dataname in enumerate(sorted(full_pointcloud_data)):
            ax = fig.add_subplot(1, nSpheres, ii + 1, projection='3d')
            noisy_data = full_pointcloud_data[dataname]
            nPts = len(noisy_data) // 2

            ax.scatter(noisy_data[0:nPts, 0], noisy_data[0:nPts, 1],
                       noisy_data[0:nPts, 2], alpha=alphas[0], color='orange')
            ax.scatter(noisy_data[nPts:2 * nPts, 0], noisy_data[nPts:2 * nPts, 1],
                       noisy_data[nPts:2 * nPts, 2], alpha=alphas[1], color='blue')
            ax.set_title(point_labels[ii])

        plt.savefig(f"{result_folder}/point_could_per_layer.png")
        plt.close("all")


    @staticmethod
    def save_heatmap(distance_matrix, title):
        ax = sns.heatmap(distance_matrix)
        fig = ax.get_figure()
        fig.savefig(title)
        plt.close("all")

    @staticmethod
    def sort_all_distances(all_distances, indicator_index):
        idxs = list(range(len(all_distances[0])))
        indicator_list = all_distances[indicator_index]

        sorted_by_Xs = sorted(
            list(zip(indicator_list, idxs)))

        size = len(all_distances)
        result_all_dists = [list() for i in range(size)]

        for _, id in sorted_by_Xs:
            for i in range(size):
                result_all_dists[i].append(all_distances[i][id])

        del all_distances
        del indicator_list
        del idxs
        del sorted_by_Xs

        return result_all_dists

    @staticmethod
    def save_distance_curves(distance_matrices, labels, ylabel, title):
        '''
        ax = sns.heatmap(distance_matrices)
        fig = ax.get_figure()
        fig.savefig(title)
        plt.close("all")
        '''

        fig, ax = plt.subplots()
        for id, D in enumerate(distance_matrices):
            D_row = D
            size = len(D_row)
            x = np.arange(size)
            ax.plot(x, D_row, label=labels[id])

        ax.set_xlabel("i-th persistence diagram")
        ax.set_ylabel(ylabel)
        plt.legend()
        plt.savefig(title)
        plt.close("all")

    @staticmethod
    def save_norm_curves(norm_curves, angles, max_angles, labels, xlabel, ylabel, title):
        fig, ax = plt.subplots()
        if norm_curves is None or len(norm_curves) == 0:
            return
        id = 0
        Eij = norm_curves[id]
        size = len(Eij)

        Xs = range(0,size)
        ax.plot(Xs, Eij, label=labels[id])
        max_val = max(Eij)
        ax.plot([max_angles[id], max_angles[id]], [0, max_val], label="maximal alpha")

        ax.set_xticks(range(0, size))
        ticklabels = ["" for i in range(size)]
        ticklabels[max_angles[id]] = max_angles[id]
        ax.set_xticklabels(ticklabels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.legend()
        plt.savefig(title)
        plt.close("all")

    @staticmethod
    def save_diagrams_as_images(diags_filename, diags):
        ax = gudhi.plot_persistence_barcode(diags, legend=True)
        ax.set_title("Barcode")
        plt.savefig(f"{diags_filename}.barcode.png")
        plt.close(plt.gcf())
        gudhi.plot_persistence_diagram(diags, legend=True)
        ax.set_title("Barcode")
        plt.savefig(f"{diags_filename}.pdiagram.png")
        plt.close(plt.gcf())

import time
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV, KFold
from sklearn import neighbors
import numpy as np
from utilities.distance_matrix_helper import DistanceMatrixHelper

class KNearestNeighborHelper(object):
    def __init__(self, exp, train_flag=True):
        self.exp = exp
        self.y_train = None
        self.y_test = None
        self.classes = None
        self.train_choice = train_flag

    def set_train_sets(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distances(self, X_test, dtypes = None, fold_no=0):
        # compare distance between test and train
        self.exp.compute_distance_matrices(dtypes=dtypes,
                                           train=self.X_train,
                                           test=X_test,
                                           fold_no=fold_no)

        return self.exp.distance_collection

    def predict_labels(self, dists, k=1):
        if self.y_train is None or len(self.y_train) == 0:
            return

        if type(dists) == list:
            '''
            In the case where we compute distances in all homology groups
            we will have an array of distance values. So we must combine them.
            '''
            dists = np.sum(dists, axis=0)
        num_test = dists.shape[0]

        y_pred = np.zeros(num_test)
        for i in range(num_test):

            sorted_dist = np.argsort(dists[i])
            closest_y = list(self.y_train[sorted_dist[0:k]])
            y_pred[i] = (np.argmax(np.bincount(closest_y)))

        return y_pred

    def perform_cv_on_knn(self, dtype, result_file):
        try:
            result_file.write("----- kNN Classifier (perform_grid_search_on_knn) -------\n")
            k_to_accuracies = {}
            dist_name = DistanceMatrixHelper.distance_names[dtype]
            for k in range(3, 100, 2):
                kf = KFold(n_splits=10, random_state=123, shuffle=True)
                kf.get_n_splits(self.exp.train_pdiags)
                k_to_accuracies[k] = {}
                fold_num = 0
                for train_idxs, test_idxs in kf.split(self.exp.train_pdiags):
                    fold_num += 1
                    X_train, X_test = self.exp.retrieve_train_test_pdiags(train_idxs, test_idxs)
                    y_train, y_test = self.original_train[train_idxs], self.original_train[test_idxs]
                    num_test = len(y_test)
                    self.set_train_sets(X_train, y_train)

                    dists = self.compute_distances(X_test, dtypes=[dtype], fold_no=fold_num)
                    distances = []
                    labels = []
                    if not DistanceMatrixHelper.is_common_type(dtype):
                        for item in dists[dtype]:
                            distances.append(dists[dtype][item])
                            labels.append(f"{dist_name}{item}")
                    else:
                        distances.append(dists[dtype])
                        labels.append(dist_name)

                    for id, dist in enumerate(distances):
                        if labels[id] not in k_to_accuracies[k]:
                            k_to_accuracies[k].update({labels[id]: []})

                        y_test_pred = self.predict_labels(dist, k)

                        num_correct = np.sum(y_test_pred == y_test)
                        accuracy = float(num_correct) / num_test
                        print(f"Fold {fold_num}- {labels[id]} Got {num_correct} / {num_test} correct => accuracy: {accuracy}\n")
                        result_file.write(
                            f"Fold {fold_num} - {labels[id]} Got {num_correct} / {num_test} correct => accuracy: {accuracy}\n")
                        k_to_accuracies[k][labels[id]].append(accuracy)
                # obtain accuracy avg and std per k
            print(f"\n\nCompute average and std for all k\n\n")
            result_file.write(f"\n\nCompute average and std for k={k}\n")
            for k in k_to_accuracies:
                print(f"Report k={k}:\n")
                result_file.write(f"Report k={k}:\n")
                for label_dist in k_to_accuracies[k]:
                    acc_avg = np.average(k_to_accuracies[k][label_dist])
                    acc_std = np.std(k_to_accuracies[k][label_dist])

                    print(f"==> {label_dist} acc (macro_avg: {acc_avg}, std: {acc_std})\n")
                    result_file.write(f"==> {label_dist} acc (macro_avg: {acc_avg}, std: {acc_std})\n")

        except Exception as e:
            print(e)
            result_file.close()

    def perform_clfr(self, dtype, result_file):
        k_choices = range(1,30)

        X_train, X_test, y_train, y_test = self.exp.get_train_test_data()
        self.set_train_sets(X_train, y_train)
        dists = self.compute_distances(X_test, dtypes=[dtype])
        dist_name = DistanceMatrixHelper.distance_names[dtype]

        k_to_accuracies = {}

        for k in k_choices:
            k_to_accuracies[k] = {}
            num_test = len(y_test)

            distances = []
            labels = []
            if not DistanceMatrixHelper.is_common_type(dtype):
                for item in dists[dtype]:
                    distances.append(dists[dtype][item])
                    labels.append(f"{dist_name}{item}")
            else:
                distances.append(dists[dtype])
                labels.append(dist_name)

            for id, dist in enumerate(distances):
                if labels[id] not in k_to_accuracies[k]:
                    k_to_accuracies[k].update({labels[id]: []})

                y_test_pred = self.predict_labels(dist, k)

                num_correct = np.sum(y_test_pred == y_test)
                accuracy = float(num_correct) / num_test
                k_to_accuracies[k][labels[id]].append(accuracy)

        print(f"\n\nCompute average and std for all k\n\n")
        result_file.write(f"\n\nCompute average and std for k={k}\n")
        for k in k_to_accuracies:
            print(f"Report k={k}:\n")
            result_file.write(f"Report k={k}:\n")
            for label_dist in k_to_accuracies[k]:
                acc_avg = np.average(k_to_accuracies[k][label_dist])
                acc_std = np.std(k_to_accuracies[k][label_dist])

                print(f"==> {label_dist} acc (macro_avg: {acc_avg}, std: {acc_std})\n")
                result_file.write(f"==> {label_dist} acc (macro_avg: {acc_avg}, std: {acc_std})\n")

    def execute(self):
        if self.train_choice:
            self.train()
            return

        self.original_train, self.original_test = self.exp.get_train_test_label()
        self.classes = np.unique(self.original_train)

        distance_type = DistanceMatrixHelper.get_all_distance_types()
        total_size = self.exp.get_total_size()
        file_name = time.strftime(
            "{0}%y.%m.%d__%H.%M.%S_exec_KNN_result_file_{1}.txt".format(self.exp.td_path,
                                                                            total_size))
        result_file = open(file_name, "w")
        try:
            for dtype in distance_type:
                result_file.write("\n################################################################\n")
                print("\n################################################################\n")
                result_file.write(f"Starting {DistanceMatrixHelper.distance_names[dtype]}\n")
                print(f"Starting {DistanceMatrixHelper.distance_names[dtype]}\n")
                result_file.write("################################################################\n")
                print("################################################################\n")
                self.perform_clfr(dtype, result_file)

            result_file.close()
            exit(0)
        except Exception as e:
            print(e)
            result_file.close()

    def train(self):
        '''
        In this method we perform a randomized search considering all topological distances.
        Note that we save the distance matrices to avoid recomputing things as much as possible.

        These matrices are stored in a subfolder calles distances. See DistanceMatricesHelper for details
        '''

        self.original_train, self.original_test = self.exp.get_train_test_data()
        self.classes = np.unique(self.original_train)

        distance_type = DistanceMatrixHelper.get_all_distance_types()

        file_name = time.strftime(
            "{0}%y.%m.%d__%H.%M.%S_KNN_result_file.txt".format(self.exp.td_path))
        result_file = open(file_name, "w")
        try:
            for dtype in distance_type:
                result_file.write("\n################################################################\n")
                print("\n################################################################\n")
                result_file.write(f"Starting {DistanceMatrixHelper.distance_names[dtype]}\n")
                print(f"Starting {DistanceMatrixHelper.distance_names[dtype]}\n")
                result_file.write("################################################################\n")
                print("################################################################\n")

                self.perform_cv_on_knn(dtype, result_file)
            result_file.close()
        except Exception as e:
            print(e)
            result_file.close()

    def perform_randomized_search(self, distance, y, result_file):
        result_file.write("----- kNN Classifier (perform_grid_search_on_knn) -------\n")
        k_range = list(range(3, 32))
        weight_options = ['uniform', 'distance']
        param_grid = {'metric': ('precomputed',),
                      'n_neighbors': k_range,
                      'weights': weight_options}

        best_params = []
        best_scores = []
        best_estimator = []

        model = neighbors.KNeighborsClassifier()

        rand = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')
        rand.fit(distance, y)
        best_scores.append(rand.best_score_)
        best_estimator.append(rand.best_estimator_)
        best_params.append(rand.best_params_)

        print(best_scores)
        result_file.write(f"\nSCORES: {str(best_scores)}\n ESTIMATOR: {str(best_estimator)} PARAMS: {str(best_params)}")


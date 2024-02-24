from sklearn.experimental import enable_halving_search_cv  # noqa
from experiments.sl_experiments import SLExperiment
from utilities.knn_classifier_helper import KNearestNeighborHelper


class FashionExperiment(SLExperiment):
    def __init__(self, data_processor,
                 p = 2,
                 no_train_samples = None,
                 no_test_samples = None,
                 save_diags = False):
        super(FashionExperiment, self).__init__(data_processor,
                                                p,
                                                no_train_samples,
                                                no_test_samples,
                                                save_diags)
        self.save_all = False

    def execute(self):
        knn = KNearestNeighborHelper(self, train_flag=False)
        knn.execute()

if __name__ == '__main__':
    import traceback
    import os

    global_path = os.environ["PWD"]
    print(global_path)

    try:
        from utilities.data_processors_helper import DataProcessorFactory

        overall_path = "results/SupervisedLearningApp/"

        fashion_processor = DataProcessorFactory.get_data_processor(overall_path=overall_path,
                                                                  data_type=DataProcessorFactory.FASHION_HOG)
        fashion_processor.execute()
        fashion = FashionExperiment(data_processor=fashion_processor,
                                    p=2,
                                    no_train_samples=100,
                                    no_test_samples=10)
        # fashion.compute_distance_matrices()
        fashion.execute()
    except Exception as e:
        print(e)
        traceback.print_exc()


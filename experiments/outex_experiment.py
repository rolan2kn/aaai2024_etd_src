from experiments.sl_experiments import SLExperiment
from utilities.knn_classifier_helper import KNearestNeighborHelper


class OutexExperiment(SLExperiment):
    def __init__(self, data_processor,
                 p = 2,
                 no_train_samples = None,
                 no_test_samples = None,
                 save_diags = False):
        super(OutexExperiment, self).__init__(data_processor, p, no_train_samples, no_test_samples, save_diags)
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

        outex_processor = DataProcessorFactory.get_data_processor(overall_path=overall_path,
                                                                  data_type=DataProcessorFactory.OUTEX)
        outex_processor.execute()
        # outex = OutexExperiment(data_processor=outex_processor, p=2, no_train_samples=20, no_test_samples=600)
        outex = OutexExperiment(data_processor=outex_processor, p=2)
        outex.execute()
    except Exception as e:
        print(e)
        traceback.print_exc()


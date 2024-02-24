from applications import AutoencoderWeightTopologyApp, SupervisedLearningApp
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Define folder to save results",
                        type=str)
    parser.add_argument("-a", "--application", help="Define the desired app to run 1: Outer dataset 2: Autoencoder ",
                        type=int)

    args = parser.parse_args()

    overall_path = "results/"
    if args.path is not None:
        overall_path = args.path

    AE, SL, ALL = range(3)
    # app = ALL
    app = SL
    # app = AE
    if args.application is not None:
        app = args.application

    if app in (SL, ALL):
        supervised_learning_app = SupervisedLearningApp(overall_path=overall_path)
        supervised_learning_app.execute()
    if app in (AE, ALL):
        ae_wt = AutoencoderWeightTopologyApp(overall_path=overall_path)
        ae_wt.execute()


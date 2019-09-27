import mlsurvey as mls


def main():
    mlw = mls.workflows.SupervisedLearningWorkflow(config_directory='files/slw', logging_dir='slw')
    mlw.run()

    mlw = mls.workflows.SupervisedLearningWorkflow(config_directory='files/visualization/base',
                                                   logging_dir='visualization/base')
    mlw.run()

    mlw = mls.workflows.SupervisedLearningWorkflow(config_directory='files/visualization/blobs',
                                                   logging_dir='visualization/blobs')
    mlw.run()

    mlw = mls.workflows.SupervisedLearningWorkflow(config_directory='files/visualization/dask_de',
                                                   logging_dir='visualization/dask_de')
    mlw.run()

    mlw = mls.workflows.SupervisedLearningWorkflow(config_directory='files/visualization/germancredit',
                                                   logging_dir='visualization/germancredit')
    mlw.run()

    mlw = mls.workflows.SupervisedLearningWorkflow(config_directory='files/visualization/svc',
                                                   logging_dir='visualization/svc')
    mlw.run()

    mlw = mls.workflows.SupervisedLearningWorkflow(config_directory='files/visualize-log/directory1',
                                                   logging_dir='visualize-log/directory1')
    mlw.run()

    mlw = mls.workflows.SupervisedLearningWorkflow(config_directory='files/visualize-log/directory2',
                                                   logging_dir='visualize-log/directory2')
    mlw.run()

    mlw = mls.workflows.SupervisedLearningWorkflow(config_directory='files/visualize-log/directory3',
                                                   logging_dir='visualize-log/directory3')
    mlw.run()

    mlw = mls.workflows.SupervisedLearningWorkflow(config_directory='files/visualize-log/directory4',
                                                   logging_dir='visualize-log/directory4')
    mlw.run()


if __name__ == "__main__":
    main()

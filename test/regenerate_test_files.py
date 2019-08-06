import mlsurvey as mls


def main():
    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='files/slw')
    mlw.run()

    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='files/visualization/base')
    mlw.run()

    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='files/visualization/blobs')
    mlw.run()

    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='files/visualization/germancredit')
    mlw.run()

    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='files/visualization/svc')
    mlw.run()

    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='files/visualize-log/directory1')
    mlw.run()

    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='files/visualize-log/directory2')
    mlw.run()

    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='files/visualize-log/directory3')
    mlw.run()


if __name__ == "__main__":
    main()

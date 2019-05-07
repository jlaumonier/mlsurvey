import mlsurvey as mls


def main():
    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='test/files/slw')
    mlw.run()

    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='test/files/visualization/base')
    mlw.run()

    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='test/files/visualization/blobs')
    mlw.run()

    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='test/files/visualization/germancredit')
    mlw.run()

    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='test/files/visualization/svc')
    mlw.run()

    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='test/files/visualize-log/directory1')
    mlw.run()

    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='test/files/visualize-log/directory2')
    mlw.run()

    mlw = mls.workflows.MultipleLearningWorkflow(config_directory='test/files/visualize-log/directory3')
    mlw.run()


if __name__ == "__main__":
    main()

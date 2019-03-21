import mlsurvey as mls


def visualize(directory):
    vw = mls.workflows.VisualizationWorkflow(directory=directory)
    vw.run()


def main():
    mlw = mls.workflows.MultipleLearningWorkflow()
    mlw.run()
    for sl in mlw.slw:
        visualize(sl.log.directory)


if __name__ == "__main__":
    main()

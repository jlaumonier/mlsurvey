import mlsurvey as mls


def visualize(directory):
    vw = mls.VisualizationWorkflow(directory=directory)
    vw.run()


def main():
    mlw = mls.MultipleLearningWorkflow()
    mlw.run()
    for sl in mlw.slw:
        visualize(sl.log.directory)


if __name__ == "__main__":
    main()

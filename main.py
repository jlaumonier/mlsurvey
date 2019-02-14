import mlsurvey as mls


def visualize(directory):
    slw = mls.SupervisedLearningWorkflow()
    slw.load_data_classifier(directory)
    mls.Visualization.plot_result(slw.data_train.x, slw.data_train.y, slw.classifier, "Train", False, 1)
    plt = mls.Visualization.plot_result(slw.data_test.x, slw.data_test.y, slw.classifier, "Test", False, 2)
    print(slw.score)
    plt.show()


def main():
    mlw = mls.MultipleLearningWorkflow()
    mlw.run()
    for sl in mlw.slw:
        visualize(sl.log.directory)


if __name__ == "__main__":
    main()

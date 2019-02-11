import mlsurvey as mls


def visualize():
    log = mls.Logging()
    i = log.load_input('input.json')
    mls.Visualization.plot_data(i.x, i.y)


def main():
    slw = mls.SupervisedLearningWorkflow()
    slw.run()
    mls.Visualization.plot_result(slw.data.x, slw.data.y, slw.classifier, "test", False)
    print(slw.score)
    log = mls.Logging()
    log.save_input(slw.data_train)


if __name__ == "__main__":
    main()
    visualize()

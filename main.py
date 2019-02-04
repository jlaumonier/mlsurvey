import mlsurvey as mls


def main():
    visual = mls.Visualization()
    inpt = mls.Input()

    algorithm_family = 'knn'
    hyperparameters = {
        'n_neighbors': 15,
        'algorithm': 'auto',
        'weights': 'uniform'
    }
    algorithm = mls.Algorithm(algorithm_family, hyperparameters)
    classifier = algorithm.learn(inpt.x, inpt.y)

    visual.plot_data(inpt.x, inpt.y)
    visual.plot_result(inpt.x, classifier)


if __name__ == "__main__":
    main()

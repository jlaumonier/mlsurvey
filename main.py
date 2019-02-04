from mlsurvey import algorithm as alg, input as inp, visualization as v


def main():
    visual = v.Visualization()
    inpt = inp.Input()

    algorithm_family = 'knn'
    hyperparameters = {
        'n_neighbors': 15,
        'algorithm': 'auto',
        'weights': 'uniform'
    }
    algorithm = alg.Algorithm(algorithm_family, hyperparameters)
    classifier = algorithm.learn(inpt.x, inpt.y)

    visual.plot_data(inpt.x, inpt.y)
    visual.plot_result(inpt.x, classifier)


if __name__ == "__main__":
    main()

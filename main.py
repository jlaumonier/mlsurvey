import mlsurvey as mls


def main():

    inpt = mls.Input()
    data = mls.datasets.DataSetFactory.create_dataset("Moons")
    data.generate()
    inpt.set_data(data)

    algorithm_family = 'knn'
    hyperparameters = {
        'n_neighbors': 15,
        'algorithm': 'auto',
        'weights': 'uniform'
    }
    algorithm = mls.Algorithm(algorithm_family, hyperparameters)
    classifier = algorithm.learn(inpt.x, inpt.y)

    mls.Visualization.plot_data(inpt.x, inpt.y)
    mls.Visualization.plot_result(inpt.x, classifier)

    log = mls.Logging()
    log.save_input(inpt)


if __name__ == "__main__":
    main()

from sklearn.preprocessing import StandardScaler

import mlsurvey as mls


def visualize():
    log = mls.Logging()
    i = log.load_input('input.json')
    mls.Visualization.plot_data(i.x, i.y)


def main():
    name = 'NClassRandomClassification'
    inpt = mls.Input()
    data = mls.datasets.DataSetFactory.create_dataset(name)
    data.generate()
    data.x = StandardScaler().fit_transform(data.x)
    inpt.set_data(data)

    algorithm_family = 'knn'
    hyperparameters = {
        'n_neighbors': 15,
        'algorithm': 'auto',
        'weights': 'uniform'
    }
    algorithm = mls.Algorithm(algorithm_family, hyperparameters)
    classifier = algorithm.learn(inpt.x, inpt.y)
    mls.Visualization.plot_result(inpt.x, inpt.y, classifier, name, True)
    log = mls.Logging()
    log.save_input(inpt)


if __name__ == "__main__":
    main()
    visualize()

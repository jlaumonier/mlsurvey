from sklearn.preprocessing import StandardScaler

import mlsurvey as mls


def visualize():
    log = mls.Logging()
    i = log.load_input('input.json')
    mls.Visualization.plot_data(i.x, i.y)


def main():
    config = mls.Config()
    dataset_name = config.data.learning_process.input
    inpt = mls.Input()
    data = mls.datasets.DataSetFactory.create_dataset(config.data.datasets[dataset_name].type)
    data_params = config.data.datasets[dataset_name].parameters
    data.set_generation_parameters(data_params)
    data.generate()
    data.x = StandardScaler().fit_transform(data.x)
    inpt.set_data(data)

    algorithm_name = config.data.learning_process.algorithm
    algorithm_family = config.data.algorithms[algorithm_name].algorithm_family
    hyperparameters = config.data.algorithms[algorithm_name].hyperparameters
    algorithm = mls.Algorithm(algorithm_family, hyperparameters)
    classifier = algorithm.learn(inpt.x, inpt.y)
    mls.Visualization.plot_result(inpt.x, inpt.y, classifier, dataset_name, False)
    log = mls.Logging()
    log.save_input(inpt)


if __name__ == "__main__":
    main()
    visualize()

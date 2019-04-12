import unittest

import mlsurvey as mls


class TestAlgorithm(unittest.TestCase):

    def test_init_parameters_should_be_initialized_for_knn(self):
        config = {
            'algorithm-family': 'module.Test',
            'hyperparameters': {
                'n_neighbors': 3,
                'algorithm': 'auto',
                'weights': 'uniform'
            }
        }
        algo = mls.models.Algorithm(config)
        self.assertEqual(config['algorithm-family'], algo.algorithm_family)
        self.assertEqual(config['hyperparameters']['n_neighbors'], algo.hyperparameters['n_neighbors'])
        self.assertEqual(config['hyperparameters']['algorithm'], algo.hyperparameters['algorithm'])
        self.assertEqual(config['hyperparameters']['weights'], algo.hyperparameters['weights'])

    def test_init_no_algorithm_family_parameter_should_not_init(self):
        config = {
            'toto': 'module.Test',
            'hyperparameters': {
                'n_neighbors': 3,
                'algorithm': 'auto',
                'weights': 'uniform'
            }
        }
        algo = None
        try:
            algo = mls.models.Algorithm(config)
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertIsNone(algo)

    def test_init_no_hyperparameters_parameter_should_not_init(self):
        config = {
            'algorithm-family': 'module.Test',
            'toto': {
                'n_neighbors': 3,
                'algorithm': 'auto',
                'weights': 'uniform'
            }
        }
        algo = None
        try:
            algo = mls.models.Algorithm(config)
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertIsNone(algo)

    def test_learn_generic_algorithm_knn_should_learn(self):
        config = {
            'algorithm-family': 'sklearn.neighbors.KNeighborsClassifier',
            'hyperparameters': {
                'n_neighbors': 3,
                'algorithm': 'auto',
                'weights': 'uniform'
            }
        }
        dataset = mls.datasets.DataSetFactory.create_dataset('make_circles')
        data = mls.models.Data()
        data.x, data.y = dataset.generate()
        algo = mls.models.Algorithm(config)
        classif = algo.learn(data.x, data.y)
        score = classif.score(data.x, data.y)
        self.assertEqual(1.0, score)

    def test_learn_generic_algorithm_unknown_algorithm(self):
        """
        :test : mlsurvey.models.Algorithm.learn()
        :condition : unknown algorithm
        :main_result : raise ConfigError
        """
        config = {
            'algorithm-family': 'sklearn.neighbors.UnknownClass',
            'hyperparameters': {
                'n_neighbors': 3,
                'algorithm': 'auto',
                'weights': 'uniform'
            }
        }
        dataset = mls.datasets.DataSetFactory.create_dataset('make_circles')
        data = mls.models.Data()
        data.x, data.y = dataset.generate()
        algo = mls.models.Algorithm(config)
        classif = None
        try:
            classif = algo.learn(data.x, data.y)
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertIsNone(classif)

    def test_to_dict_algo_saved(self):
        config = {
            'algorithm-family': 'sklearn.neighbors.KNeighborsClassifier',
            'hyperparameters': {
                'n_neighbors': 3,
                'algorithm': 'auto',
                'weights': 'uniform'
            }
        }
        algo = mls.models.Algorithm(config)
        d = algo.to_dict()
        self.assertEqual(config, d)

    def test_from_dict_algo_loaded(self):
        config = {
            'algorithm-family': 'module.Test',
            'hyperparameters': {
                'n_neighbors': 3,
                'algorithm': 'auto',
                'weights': 'uniform'
            }
        }
        algo = mls.models.Algorithm.from_dict(config)
        self.assertEqual(config['algorithm-family'], algo.algorithm_family)
        self.assertEqual(config['hyperparameters']['n_neighbors'], algo.hyperparameters['n_neighbors'])
        self.assertEqual(config['hyperparameters']['algorithm'], algo.hyperparameters['algorithm'])
        self.assertEqual(config['hyperparameters']['weights'], algo.hyperparameters['weights'])

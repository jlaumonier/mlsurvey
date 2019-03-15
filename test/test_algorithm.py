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
        algo = mls.Algorithm(config)
        self.assertEqual('module.Test', algo.algorithm_family)
        self.assertEqual(config['hyperparameters']['n_neighbors'], algo.hyperparameters['n_neighbors'])
        self.assertEqual(config['hyperparameters']['algorithm'], algo.hyperparameters['algorithm'])
        self.assertEqual(config['hyperparameters']['weights'], algo.hyperparameters['weights'])

    def test_learn_generic_algorithm_knn_should_learn(self):
        config = {
            'algorithm-family': 'sklearn.neighbors.KNeighborsClassifier',
            'hyperparameters': {
                'n_neighbors': 3,
                'algorithm': 'auto',
                'weights': 'uniform'
            }
        }
        data = mls.datasets.DataSetFactory.create_dataset('make_circles')
        data.generate()
        algo = mls.Algorithm(config)
        classif = algo.learn(data.x, data.y)
        score = classif.score(data.x, data.y)
        self.assertEqual(1.0, score)

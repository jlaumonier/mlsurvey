import unittest

import mlsurvey as mls


class TestAlgorithm(unittest.TestCase):

    def test_init_parameters_should_be_initialized_for_knn(self):
        algorithm_family = 'Test'
        hyperparameters = {
            'n_neighbors': 3,
            'algorithm': 'auto',
            'weights': 'uniform'
        }
        algo = mls.Algorithm(algorithm_family, hyperparameters)
        self.assertEqual(algorithm_family, algo.algorithm_family)
        self.assertEqual(hyperparameters['n_neighbors'], algo.hyperparameters['n_neighbors'])
        self.assertEqual(hyperparameters['algorithm'], algo.hyperparameters['algorithm'])
        self.assertEqual(hyperparameters['weights'], algo.hyperparameters['weights'])


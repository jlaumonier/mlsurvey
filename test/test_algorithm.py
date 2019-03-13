import unittest

import mlsurvey as mls


class TestAlgorithm(unittest.TestCase):

    def test_init_parameters_should_be_initialized_for_knn(self):
        config = {
            'algorithm-family': 'Test',
            'hyperparameters': {
                'n_neighbors': 3,
                'algorithm': 'auto',
                'weights': 'uniform'
            }
        }
        algo = mls.Algorithm(config)
        self.assertEqual(config['algorithm-family'], algo.algorithm_family)
        self.assertEqual(config['hyperparameters']['n_neighbors'], algo.hyperparameters['n_neighbors'])
        self.assertEqual(config['hyperparameters']['algorithm'], algo.hyperparameters['algorithm'])
        self.assertEqual(config['hyperparameters']['weights'], algo.hyperparameters['weights'])

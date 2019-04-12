import unittest

import numpy as np

import mlsurvey as mls


class TestEvaluationFairness(unittest.TestCase):

    def test_init_should_init(self):
        evf = mls.models.EvaluationFairness()
        self.assertIsInstance(evf.probability, np.ndarray)

    def test_to_dict(self):
        evf = mls.models.EvaluationFairness()
        evf.probability = np.array([0.7, 0.3])
        expected = {'type': 'EvaluationFairness', 'probability': [0.7, 0.3]}
        result = evf.to_dict()
        self.assertDictEqual(expected, result)

    def test_from_dict(self):
        source = {'type': 'EvaluationFairness', 'probability': [0.7, 0.3]}
        evs = mls.models.EvaluationFairness()
        evs.from_dict(source)
        self.assertListEqual(evs.probability.tolist(), [0.7, 0.3])

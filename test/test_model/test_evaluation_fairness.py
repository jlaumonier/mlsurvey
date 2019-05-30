import unittest

import numpy as np

import mlsurvey as mls


class TestEvaluationFairness(unittest.TestCase):

    def test_init_should_init(self):
        evf = mls.models.EvaluationFairness()
        self.assertIsInstance(evf.probability, np.ndarray)
        self.assertEqual(0, evf.demographic_parity)

    def test_to_dict(self):
        evf = mls.models.EvaluationFairness()
        evf.probability = np.array([0.7, 0.3])
        evf.demographic_parity = 0.33
        expected = {'type': 'EvaluationFairness', 'probability': [0.7, 0.3], 'demographic_parity': 0.33}
        result = evf.to_dict()
        self.assertDictEqual(expected, result)

    def test_from_dict(self):
        source = {'type': 'EvaluationFairness', 'probability': [0.7, 0.3], 'demographic_parity': 0.33}
        evs = mls.models.EvaluationFairness()
        evs.from_dict(source)
        self.assertListEqual(evs.probability.tolist(), [0.7, 0.3])
        self.assertEqual(0.33, evs.demographic_parity)

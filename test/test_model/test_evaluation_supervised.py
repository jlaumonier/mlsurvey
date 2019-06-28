import unittest

import numpy as np

import mlsurvey as mls


class TestEvaluationSupervised(unittest.TestCase):

    def test_init_should_init(self):
        evs = mls.models.EvaluationSupervised()
        self.assertEqual(evs.score, 0.0)
        self.assertIsInstance(evs.confusion_matrix, np.ndarray)
        self.assertIsNone(evs.sub_evaluation)

    def test_to_dict(self):
        evs = mls.models.EvaluationSupervised()
        evs.score = 0.85
        evs.confusion_matrix = np.array([[1, 2, 3],
                                         [4, 5, 6],
                                         [7, 8, 9]])
        expected = {'type': 'EvaluationSupervised',
                    'score': evs.score,
                    'confusion_matrix': evs.confusion_matrix.tolist()}
        result = evs.to_dict()
        self.assertDictEqual(expected, result)

    def test_to_dict_with_sub_evaluation(self):
        evs = mls.models.EvaluationSupervised()
        evs.score = 0.85
        evs.confusion_matrix = np.array([[1, 2, 3],
                                         [4, 5, 6],
                                         [7, 8, 9]])
        evf = mls.models.EvaluationFairness()
        evf.probability = np.array([0.7, 0.3])
        evf.demographic_parity = 0.33
        evs.sub_evaluation = evf
        expected_sub_eval = evf.to_dict()
        expected = {'type': 'EvaluationSupervised',
                    'score': evs.score,
                    'confusion_matrix': evs.confusion_matrix.tolist(),
                    'sub_evaluation': expected_sub_eval}
        result = evs.to_dict()
        self.assertDictEqual(expected, result)

    def test_from_dict(self):
        expected_confusion_matrix = np.array([[1, 2, 3],
                                              [4, 5, 6],
                                              [7, 8, 9]])
        source = {'type': 'EvaluationSupervised',
                  'score': 0.85,
                  'confusion_matrix': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
        evs = mls.models.EvaluationSupervised()
        evs.from_dict(source)
        self.assertEqual(0.85, evs.score)
        np.testing.assert_array_equal(expected_confusion_matrix, evs.confusion_matrix)

    def test_from_dict_with_sub_evaluation(self):
        expected_confusion_matrix = np.array([[1, 2, 3],
                                              [4, 5, 6],
                                              [7, 8, 9]])
        expected_sub_eval_probability = np.array([0.7, 0.3])
        expected_sub_eval_demographic_parity = 0.33
        source = {'type': 'EvaluationSupervised',
                  'score': 0.85,
                  'confusion_matrix': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                  'sub_evaluation': {'type': 'EvaluationFairness',
                                     'probability': expected_sub_eval_probability,
                                     'demographic_parity': expected_sub_eval_demographic_parity}
                  }
        evs = mls.models.EvaluationSupervised()
        evs.from_dict(source)
        self.assertEqual(0.85, evs.score)
        np.testing.assert_array_equal(expected_confusion_matrix, evs.confusion_matrix)
        np.testing.assert_array_equal(evs.sub_evaluation.probability, expected_sub_eval_probability)
        self.assertEqual(expected_sub_eval_demographic_parity, evs.sub_evaluation.demographic_parity)

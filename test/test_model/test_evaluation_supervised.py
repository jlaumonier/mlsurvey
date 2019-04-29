import unittest

import numpy as np

import mlsurvey as mls


class TestEvaluationSupervised(unittest.TestCase):

    def test_init_should_init(self):
        evs = mls.models.EvaluationSupervised()
        self.assertEqual(evs.score, 0.0)
        self.assertIsInstance(evs.confusion_matrix, np.ndarray)

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

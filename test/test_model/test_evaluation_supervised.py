import unittest

import mlsurvey as mls


class TestEvaluationSupervised(unittest.TestCase):

    def test_init_should_init(self):
        evs = mls.models.EvaluationSupervised()
        self.assertEqual(evs.score, 0.0)

    def test_to_dict(self):
        evs = mls.models.EvaluationSupervised()
        evs.score = 0.85
        expected = {'type': 'EvaluationSupervised', 'score': evs.score}
        result = evs.to_dict()
        self.assertDictEqual(expected, result)

    def test_from_dict(self):
        source = {'type': 'EvaluationSupervised', 'score': 0.85}
        evs = mls.models.EvaluationSupervised()
        evs.from_dict(source)
        self.assertEqual(0.85, evs.score)

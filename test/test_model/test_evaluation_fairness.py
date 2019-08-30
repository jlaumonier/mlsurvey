import unittest

import mlsurvey as mls


class TestEvaluationFairness(unittest.TestCase):

    def test_init_should_init(self):
        evf = mls.models.EvaluationFairness()
        self.assertEqual(0, evf.demographic_parity)
        self.assertEqual(None, evf.equal_opportunity)
        self.assertEqual(None, evf.statistical_parity)
        self.assertEqual(None, evf.average_equalized_odds)
        self.assertEqual(None, evf.disparate_impact_rate)

    def test_to_dict(self):
        evf = mls.models.EvaluationFairness()
        evf.demographic_parity = 0.33
        evf.equal_opportunity = 0.5
        evf.statistical_parity = 0.7
        evf.average_equalized_odds = 0.8
        evf.disparate_impact_rate = 0.9
        expected = {'type': 'EvaluationFairness',
                    'demographic_parity': 0.33,
                    'equal_opportunity': 0.5,
                    'statistical_parity': 0.7,
                    'average_equalized_odds': 0.8,
                    'disparate_impact_rate': 0.9}
        result = evf.to_dict()
        self.assertDictEqual(expected, result)

    def test_from_dict(self):
        source = {'type': 'EvaluationFairness',
                  'demographic_parity': 0.33,
                  'equal_opportunity': 0.5,
                  'statistical_parity': 0.7,
                  'average_equalized_odds': 0.8,
                  'disparate_impact_rate': 0.9}
        evs = mls.models.EvaluationFairness()
        evs.from_dict(source)
        self.assertEqual(0.33, evs.demographic_parity)
        self.assertEqual(0.5, evs.equal_opportunity)
        self.assertEqual(0.7, evs.statistical_parity)
        self.assertEqual(0.8, evs.average_equalized_odds)
        self.assertEqual(0.9, evs.disparate_impact_rate)

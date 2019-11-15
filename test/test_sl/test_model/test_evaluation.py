import unittest

import mlsurvey as mls


class TestEvaluation(unittest.TestCase):

    def test_init_should_init(self):
        ev = mls.sl.models.Evaluation()
        self.assertIsInstance(ev, mls.sl.models.Evaluation)

    def test_to_dict_should_converted(self):
        ev = mls.sl.models.Evaluation()
        expected = {'type': 'Evaluation'}
        result = ev.to_dict()
        self.assertDictEqual(result, expected)

    def test_from_dict_should_be_created(self):
        source_dict = {'type': 'Evaluation'}
        ev = mls.sl.models.Evaluation()
        ev.from_dict(source_dict)
        self.assertIsInstance(ev, mls.sl.models.Evaluation)

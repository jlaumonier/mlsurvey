import unittest

import numpy as np

import mlsurvey as mls


class TestData(unittest.TestCase):

    def test_init_data(self):
        i = mls.models.Data()
        self.assertListEqual([], i.x.tolist())
        self.assertListEqual([], i.y.tolist())

    def test_to_dict_dict_should_be_set(self):
        d = mls.models.Data()
        d.x = np.array([[1, 2], [3, 4]])
        d.y = np.array([0, 1])
        expected = {'data.x': [[1, 2], [3, 4]], 'data.y': [0, 1]}
        result = d.to_dict()
        self.assertDictEqual(expected, result)

    def test_from_dict_input_is_set(self):
        input_dict = {'data.x': [[1, 2], [3, 4]], 'data.y': [0, 1]}
        d = mls.models.Data.from_dict(input_dict)
        self.assertEqual(1, d.x[0, 0])
        self.assertEqual(0, d.y[0])
        self.assertEqual(2, d.x.shape[1])
        self.assertEqual(2, d.x.shape[0])
        self.assertEqual(2, d.y.shape[0])

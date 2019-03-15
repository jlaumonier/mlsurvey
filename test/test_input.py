import unittest

import numpy as np

import mlsurvey as mls


class TestInput(unittest.TestCase):

    def test_init_input(self):
        i = mls.Input()
        self.assertListEqual([], i.x.tolist())
        self.assertListEqual([], i.y.tolist())

    def test_set_input_from_iris_set(self):
        i = mls.Input()
        iris = mls.datasets.GenericDataSet('load_iris')
        iris.set_generation_parameters(None)
        iris.generate()
        i.set_data(iris)
        self.assertEqual(5.1, i.x[0, 0])
        self.assertEqual(0, i.y[0])
        self.assertEqual(4, i.x.shape[1])
        self.assertEqual(150, i.x.shape[0])
        self.assertEqual(150, i.y.shape[0])

    def test_to_dict_dict_should_be_set(self):
        i = mls.Input()
        i.x = np.array([[1, 2], [3, 4]])
        i.y = np.array([0, 1])
        expected = {'input.x': [[1, 2], [3, 4]], 'input.y': [0, 1]}
        result = i.to_dict()
        self.assertDictEqual(expected, result)

    def test_from_dict_input_is_set(self):
        i = mls.Input()
        d = {'input.x': [[1, 2], [3, 4]], 'input.y': [0, 1]}
        i.from_dict(d)
        self.assertEqual(1, i.x[0, 0])
        self.assertEqual(0, i.y[0])
        self.assertEqual(2, i.x.shape[1])
        self.assertEqual(2, i.x.shape[0])
        self.assertEqual(2, i.y.shape[0])

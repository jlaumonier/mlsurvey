import unittest

import mlsurvey as mls


class TestInput(unittest.TestCase):

    def test_init_input(self):
        i = mls.Input()
        self.assertListEqual([], i.x)
        self.assertListEqual([], i.y)

    def test_set_input_from_iris_set(self):
        i = mls.Input()
        iris = mls.datasets.Iris()
        iris.generate()
        i.set_data(iris)
        self.assertEqual(5.1, i.x[0, 0])
        self.assertEqual(0, i.y[0])
        self.assertEqual(2, i.x.shape[1])
        self.assertEqual(150, i.x.shape[0])
        self.assertEqual(150, i.y.shape[0])

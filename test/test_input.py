import unittest

import mlsurvey.input as mls


class TestInput(unittest.TestCase):

    def test_init_2D_iris_loaded(self):
        input = mls.Input()
        self.assertEqual(5.1, input.x[0, 0])
        self.assertEqual(0, input.y[0])
        self.assertEqual(2, input.x.shape[1])
        self.assertEqual(150, input.x.shape[0])
        self.assertEqual(150, input.y.shape[0])

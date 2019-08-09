import unittest

import numpy as np

import mlsurvey as mls


class TestFairnessUtils(unittest.TestCase):

    def test_calculate_probability_calculated(self):
        x = np.array([[1, 2], [3, 4], [3, 2], [3, 3], [1, 3]])
        y = np.array([0, 1, 0, 0, 1])
        d = mls.models.Data(x=x, y=y)
        proba = mls.FairnessUtils.calculate_probability(d)
        expected_proba = np.array([3 / 5, 2 / 5])
        self.assertIsInstance(proba, np.ndarray)
        np.testing.assert_array_equal(proba, expected_proba)

    def test_calculate_all_cond_probability_calculated(self):
        x = np.array([[1, 2], [3, 4], [3, 2], [3, 3], [1, 3]])
        y = np.array([0, 1, 0, 0, 1])
        d = mls.models.Data(x=x, y=y)
        proba = mls.FairnessUtils.calculate_all_cond_probability(d)
        expected_proba = [{1: [0.5, 0.5],
                           3: [0.6666666666666666, 0.3333333333333333]},
                          {2: [1.0, 0.0],
                           3: [0.5, 0.5],
                           4: [0.0, 1.0]}]
        self.assertIsInstance(proba, list)
        self.assertListEqual(proba, expected_proba)

    def test_calculate_all_cond_probability_calculated_with_priv_class(self):
        x = np.array([[1, 2], [3, 4], [3, 2], [3, 3], [1, 3]])
        y = np.array([0, 1, 0, 0, 1])
        priv = np.array([True, False, True, False, False])
        d = mls.models.Data(x=x, y=y)
        d.add_column_in_data(priv)
        proba = mls.FairnessUtils.calculate_all_cond_probability(d)
        expected_proba = [{1: [0.5, 0.5],
                           3: [0.6666666666666666, 0.3333333333333333]},
                          {2: [1.0, 0.0],
                           3: [0.5, 0.5],
                           4: [0.0, 1.0]},
                          {False: [0.3333333333333333, 0.6666666666666666],
                           True: [1.0, 0.0]}]
        self.assertIsInstance(proba, list)
        self.assertListEqual(proba, expected_proba)

    def test_calculate_all_cond_probability_calculated_with_one_class_in_y(self):
        x = np.array([[1, 2], [3, 4], [3, 2], [3, 3], [1, 3]])
        y = np.array([0, 0, 0, 0, 0])
        d = mls.models.Data(x=x, y=y)
        proba = mls.FairnessUtils.calculate_all_cond_probability(d)
        expected_proba = [{1: [1.0],
                           3: [1.0]},
                          {2: [1.0],
                           3: [1.0],
                           4: [1.0]}]
        self.assertIsInstance(proba, list)
        self.assertListEqual(proba, expected_proba)

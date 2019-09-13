import unittest

import dask.dataframe as dd
import numpy as np
import pandas as pd

import mlsurvey as mls


class TestFairnessUtils(unittest.TestCase):

    def test_calculate_cond_probability_calculated_pandas(self):
        x = np.array([[1, 2], [3, 4], [3, 2], [3, 3], [1, 3]])
        y = np.array([0, 1, 0, 0, 1])
        data_array = np.concatenate((x, np.array([y]).T), axis=1)
        df = pd.DataFrame(data=data_array)
        data = mls.models.Data(df)
        proba1 = mls.FairnessUtils.calculate_cond_probability(data, [('target', 0)], [('C0', 1)])
        expected_proba1 = 0.5
        proba2 = mls.FairnessUtils.calculate_cond_probability(data, [('target', 1)], [('C0', 1)])
        expected_proba2 = 0.5
        proba3 = mls.FairnessUtils.calculate_cond_probability(data, [('target', 1)], [('C0', 3)])
        expected_proba3 = 0.3333333333333333
        self.assertEqual(proba1, expected_proba1)
        self.assertEqual(proba2, expected_proba2)
        self.assertEqual(proba3, expected_proba3)

    def test_calculate_cond_probability_calculated_dask(self):
        x = np.array([[1, 2], [3, 4], [3, 2], [3, 3], [1, 3]])
        y = np.array([0, 1, 0, 0, 1])
        data_array = np.concatenate((x, np.array([y]).T), axis=1)
        df = dd.from_array(data_array)
        data = mls.models.Data(df)
        proba1 = mls.FairnessUtils.calculate_cond_probability(data, [('target', 0)], [('C0', 1)])
        expected_proba1 = 0.5
        proba2 = mls.FairnessUtils.calculate_cond_probability(data, [('target', 1)], [('C0', 1)])
        expected_proba2 = 0.5
        proba3 = mls.FairnessUtils.calculate_cond_probability(data, [('target', 1)], [('C0', 3)])
        expected_proba3 = 0.3333333333333333
        self.assertEqual(proba1, expected_proba1)
        self.assertEqual(proba2, expected_proba2)
        self.assertEqual(proba3, expected_proba3)

    def test_calculate_cond_probability_calculated_two_givens(self):
        x = np.array([[1, 2], [3, 4], [3, 2], [3, 3], [1, 3], [1, 2]])
        y = np.array([0, 1, 0, 0, 1, 1])
        data_array = np.concatenate((x, np.array([y]).T), axis=1)
        df = pd.DataFrame(data=data_array)
        data = mls.models.Data(df)
        proba1 = mls.FairnessUtils.calculate_cond_probability(data, [('target', 0)], [('C0', 1), ('C1', 2)])
        expected_proba1 = 0.5
        self.assertEqual(proba1, expected_proba1)

    def test_calculate_cond_probability_calculated_two_ofs(self):
        x = np.array([[1, 2], [3, 4], [3, 2], [3, 3], [1, 3], [1, 2]])
        y = np.array([0, 1, 0, 0, 1, 1])
        data_array = np.concatenate((x, np.array([y]).T), axis=1)
        df = pd.DataFrame(data=data_array)
        data = mls.models.Data(df)
        proba1 = mls.FairnessUtils.calculate_cond_probability(data, [('target', 0), ('C0', 1)], [('C1', 2)])
        expected_proba1 = 0.3333333333333333
        self.assertEqual(proba1, expected_proba1)

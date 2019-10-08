import unittest

import numpy as np
import pandas as pd

import mlsurvey as mls


class TestData(unittest.TestCase):

    def test_to_dict_dict_should_be_set(self):
        """
        :test : mlsurvey.model.Data.to_dict()
        :condition : x,y, y_pred data are filled.
                    Use DataPandas but should work also with Dask
        :main_result : the dictionary generated is the same as expected
        """
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([0, 1])
        y_pred = np.array([1, 0])
        data_array = np.concatenate((x, np.array([y]).T, np.array([y_pred]).T), axis=1)
        df = pd.DataFrame(data=data_array)
        data = mls.models.DataPandas(df, df_contains='xyypred')
        expected = {'df_contains': 'xyypred',
                    'y_col_name': 'target',
                    'y_pred_col_name': 'target_pred'}
        result = data.to_dict()
        self.assertDictEqual(expected, result)

    def test_from_dict_df_empty(self):
        """
        :test : mlsurvey.model.DataPandas.from_dict()
        :condition : the input dict is set and an empty dataframe is given.
                     Use DataPandas but should work also with Dask
        :main_result : a ModelError occurs
        """
        df = pd.DataFrame(data=np.array([]))
        d = None
        input_dict = {'df_contains': 'xyypred',
                      'y_col_name': 'target',
                      'y_pred_col_name': 'target_pred'}
        try:
            d = mls.models.DataPandas.from_dict(input_dict, df)
            self.assertTrue(False)
        except mls.exceptions.ModelError:
            self.assertIsNone(d)
            self.assertTrue(True)

    def test_from_dict_dict_empty(self):
        """
        :test : mlsurvey.model.Data.from_dict()
        :condition : the input dict does not contains all keys and an full dataframe is given
        :main_result : a ModelError occurs
        """
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        y_pred = np.array([1, 0])
        data_array = np.concatenate((x, np.array([y]).T, np.array([y_pred]).T), axis=1)
        df = pd.DataFrame(data=data_array)
        data = None
        input_dict = {'df_contains': 'xyypred',
                      'y_pred_col_name': 'target_pred'}
        try:
            data = mls.models.DataPandas.from_dict(input_dict, df)
            self.assertTrue(False)
        except mls.exceptions.ModelError:
            self.assertIsNone(data)
            self.assertTrue(True)


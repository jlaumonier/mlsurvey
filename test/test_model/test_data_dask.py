import unittest

import dask.dataframe as dd
import numpy as np
import pandas as pd

import mlsurvey as mls


class TestDataPanda(unittest.TestCase):

    def test_init_data_all_empty_dask(self):
        """
        :test : mlsurvey.model.DataDask()
        :condition : input Dask Dataframe is empty
        :main_result : model error is raise
        """
        df = dd.from_array(np.array([]))
        try:
            _ = mls.models.DataDask(df)
            self.assertTrue(False)
        except mls.exceptions.ModelError:
            self.assertTrue(True)

    def test_init_data_x_y_without_column_name_dask(self):
        """
        :test : mlsurvey.model.DataDask()
        :condition : dask dataframe created with x and y data as numpy arrays
        :main_result : x and y data are set
        """
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        data_array = np.concatenate((x, np.array([y]).T), axis=1)
        df = dd.from_array(data_array)
        data = mls.models.DataDask(df, y_col_name='target')
        expected_column_name = ['C0', 'C1', 'target']
        np.testing.assert_array_equal(x, data.x.compute())
        np.testing.assert_array_equal(y, data.y.compute())
        self.assertEqual('xy', data.df_contains)
        self.assertEqual(-1, data.max_x_column)
        self.assertEqual('target', data.y_col_name)
        self.assertEqual('target_pred', data.y_pred_col_name)
        self.assertListEqual(expected_column_name, list(data.df.columns))
        self.assertIsInstance(data.df, dd.DataFrame)

    def test_init_data_x_y_with_column_name_dask(self):
        """
        :test : mlsurvey.model.DataDask()
        :condition : dask dataframe created with x and y data as numpy arrays. Column are given
        :main_result : x and y data are set
        """
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        data_array = np.concatenate((x, np.array([y]).T), axis=1)
        expected_column_name = ['Col1', 'Col2', 'Col3']
        df = dd.from_array(data_array, columns=expected_column_name)
        data = mls.models.DataDask(df, y_col_name='Col3')
        self.assertEqual('xy', data.df_contains)
        self.assertEqual(-1, data.max_x_column)
        self.assertEqual('Col3', data.y_col_name)
        self.assertEqual('target_pred', data.y_pred_col_name)
        np.testing.assert_array_equal(x, data.x)
        np.testing.assert_array_equal(y, data.y)
        self.assertListEqual(expected_column_name, list(data.df.columns))

    def test_init_data_x_y_ypred_dask(self):
        """
        :test : mlsurvey.model.DataDask()
        :condition : dask dataframe created with x,y and y_pred data as numpy arrays
        :main_result : x, y and y_pred data are set
        """
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        y_pred = np.array([1, 0])
        data_array = np.concatenate((x, np.array([y]).T, np.array([y_pred]).T), axis=1)
        df = dd.from_array(data_array)
        expected_column_name = ['C0', 'C1', 'target', 'target_pred']
        data = mls.models.DataDask(df, df_contains='xyypred')
        self.assertEqual('xyypred', data.df_contains)
        self.assertEqual(-2, data.max_x_column)
        self.assertEqual('target', data.y_col_name)
        self.assertEqual('target_pred', data.y_pred_col_name)
        np.testing.assert_array_equal(x, data.x)
        np.testing.assert_array_equal(y, data.y)
        np.testing.assert_array_equal(y_pred, data.y_pred)
        self.assertListEqual(expected_column_name, list(data.df.columns))

    def test_set_pred_data_data_set_dask(self):
        """
        :test : mlsurvey.model.Data.set_pred_data()
        :condition : y prediction data is given as dask dataframe
        :main_result : y prediction data is set
        """
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        data_array = np.concatenate((x, np.array([y]).T), axis=1)
        df = dd.from_array(data_array)
        data = mls.models.DataDask(df, y_col_name='target')
        y_pred = np.array([1, 0])
        # if array is not in 2d, Dask creates a Series and not a Dataframe !!!
        df_y_pred = dd.from_array(y_pred).to_frame()
        data.set_pred_data(df_y_pred)
        self.assertEqual('xyypred', data.df_contains)
        np.testing.assert_array_equal(y_pred, data.y_pred)

    def test_set_pred_data_with_colname_data_set_dask(self):
        """
        :test : mlsurvey.model.Data.set_pred_data()
        :condition : y prediction data is given as dask dataframe. y_pred column name is set into the y_pred dataframe
        :main_result : y prediction data is set. Columns are set
        """
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        data_array = np.concatenate((x, np.array([y]).T), axis=1)
        df = dd.from_array(data_array)
        data = mls.models.DataDask(df, y_col_name='target')
        y_pred = np.array([1, 0])
        df_y_pred = dd.from_array(y_pred, columns=['Col3'])
        expected_column_name = ['C0', 'C1', 'target', 'Col3']
        data.set_pred_data(df_y_pred)
        np.testing.assert_array_equal(y_pred, data.y_pred)
        self.assertEqual('xyypred', data.df_contains)
        self.assertListEqual(expected_column_name, list(data.df.columns))

    def test_set_pred_data_colname_in_func_data_set_dask(self):
        """
        :test : mlsurvey.model.Data.set_pred_data()
        :condition : y prediction data is given as dask dataframe.
                     y_pred column name is set into the set_pred_data function
        :main_result : y prediction data is set
        """
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        data_array = np.concatenate((x, np.array([y]).T), axis=1)
        df = dd.from_array(data_array)
        data = mls.models.DataDask(df, y_col_name='target')
        y_pred = np.array([1, 0])
        # if array is not in 2d, Dask creates a Series and not a Dataframe !!!
        df_y_pred = dd.from_array(y_pred).to_frame()
        expected_column_name = ['C0', 'C1', 'target', 'Col3']
        data.set_pred_data(df_y_pred, y_pred_col_name='Col3')
        np.testing.assert_array_equal(y_pred, data.y_pred)
        self.assertEqual('xyypred', data.df_contains)
        self.assertListEqual(expected_column_name, list(data.df.columns))

    def test_set_pred_data_colname_in_df_and_func_data_set_dask(self):
        """
        :test : mlsurvey.model.Data.set_pred_data()
        :condition : y prediction data is given as dask dataframe.
                     y_pred column name is set into the set_pred_data function and into the df (different)
        :main_result : y prediction data is set. column name is the parameters
        """
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        data_array = np.concatenate((x, np.array([y]).T), axis=1)
        df = dd.from_array(data_array)
        data = mls.models.DataDask(df, y_col_name='target')
        y_pred = np.array([1, 0])
        df_y_pred = dd.from_array(y_pred, columns=['FakeNameCol3'])
        expected_column_name = ['C0', 'C1', 'target', 'RealNameCol3']
        data.set_pred_data(df_y_pred, y_pred_col_name='RealNameCol3')
        np.testing.assert_array_equal(y_pred, data.y_pred)
        self.assertEqual('RealNameCol3', data.y_pred_col_name)
        self.assertEqual('xyypred', data.df_contains)
        self.assertListEqual(expected_column_name, list(data.df.columns))

    def test_set_pred_data_data_set_dask_and_pandas(self):
        """
        :test : mlsurvey.model.Data.set_pred_data()
        :condition : data is a dask dataframe whereas y prediction data is given as pandas dataframe
        :main_result : y_pred is joint into data
        """
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        data_array = np.concatenate((x, np.array([y]).T), axis=1)
        df = dd.from_array(data_array)
        data = mls.models.DataDask(df)
        y_pred = np.array([1, 0])
        df_y_pred = pd.DataFrame(data=y_pred)
        data.set_pred_data(df_y_pred)
        np.testing.assert_array_equal(y_pred, data.y_pred)
        self.assertEqual('xyypred', data.df_contains)
        self.assertIsInstance(data.df, dd.DataFrame)

    def test_add_calculated_column_column_added_dask(self):
        """
        :test : mlsurvey.model.Data.add_calculated_column()
        :condition : x,y, y_pred data are filled. dataframe is dask.
        :main_result : column and data added.
        """
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([0, 1])
        y_pred = np.array([1, 0])
        data_array = np.concatenate((x, np.array([y]).T, np.array([y_pred]).T), axis=1)
        df = dd.from_array(data_array)
        data = mls.models.DataDask(df, df_contains='xyypred')
        condition = 'x >= 2'
        on_column_name = 'C0'
        new_column_name = 'Calculated'
        data.add_calculated_column(condition, on_column_name, new_column_name)
        expected_result = [False, True]
        self.assertEqual(type(expected_result[0]), type(data.x[0, -1].compute()))
        np.testing.assert_array_equal(expected_result, data.x[:, -1])
        self.assertEqual('Calculated', data.df.columns[data.max_x_column - 1])

    def test_merge_all_should_merge_dask(self):
        """
        :test : mlsurvey.modles.Data.merge_all()
        :condition : data contains x, y and y_pred. Dataframe is dask
        :main_result : data are merge into one array
        """
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        y_pred = np.array([1, 0])
        data_array = np.concatenate((x, np.array([y]).T, np.array([y_pred]).T), axis=1)
        df = dd.from_array(data_array)
        data = mls.models.DataDask(df, df_contains='xyypred')
        expected_result = np.asarray([[1, 2, 0, 1],
                                      [3, 4, 1, 0]])
        result = data.merge_all().compute()
        np.testing.assert_array_equal(expected_result, result)
        self.assertIsInstance(result, np.ndarray)

    def test_from_dict_input_is_set_dask(self):
        """
        :test : mlsurvey.model.Data.from_dict()
        :condition : the input dict is set and a full dask df is given
        :main_result : the data is created as expected
        """
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        y_pred = np.array([1, 0])
        data_array = np.concatenate((x, np.array([y]).T, np.array([y_pred]).T), axis=1)
        df = dd.from_array(data_array)
        input_dict = {'df_contains': 'xyypred',
                      'y_col_name': 'target',
                      'y_pred_col_name': 'target_pred'}
        data = mls.models.DataDask.from_dict(input_dict, df)
        self.assertEqual('xyypred', data.df_contains)
        self.assertEqual('target', data.y_col_name)
        self.assertEqual('target_pred', data.y_pred_col_name)
        self.assertEqual(1, data.x[0, 0])
        self.assertEqual(0, data.y[0])
        self.assertEqual(1, data.y_pred[0])
        self.assertEqual(2, data.x.shape[1])
        self.assertEqual(2, data.x.shape[0])
        self.assertEqual(2, data.y.shape[0])
        self.assertEqual(2, data.y_pred.shape[0])

    def test_copy_with_new_data_dask(self):
        """
        :test : mlsurvey.modles.DataDask.copy_with_new_data()
        :condition : Dataframe is dask, x and y a set
        :main_result : new data instance is return with new data values but same columns names and other paramters
        """
        x = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        data_array = np.concatenate((x, np.array([y]).T), axis=1)
        expected_column_name = ['Col1', 'Col2', 'Col3']
        df = dd.from_array(data_array, columns=expected_column_name)
        data = mls.models.DataDask(df, df_contains='xy', y_col_name=expected_column_name[2])
        new_x = np.array([[10, 20], [30, 40]])
        new_data = data.copy_with_new_data([new_x, y])
        self.assertIsInstance(new_data.df, data.df.__class__)
        self.assertIsInstance(new_data, mls.models.DataDask)
        self.assertEqual('xy', new_data.df_contains)
        self.assertEqual('Col3', new_data.y_col_name)
        np.testing.assert_array_equal(new_x, new_data.x)
        np.testing.assert_array_equal(y, new_data.y)
        self.assertListEqual(expected_column_name, list(new_data.df.columns))

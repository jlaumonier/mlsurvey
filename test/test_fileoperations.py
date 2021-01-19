import os
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import mlsurvey as mls


class TestFileOperation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.base_directory = os.path.join(directory, '')

    def test_save_json_file_saves(self):
        directory = os.path.join(self.base_directory, 'logs/testing/')
        data = {'testA': [[1, 2], [3, 4]], 'testB': 'Text'}
        mls.FileOperation.save_dict_as_json('dict.json', directory, data)
        self.assertTrue(os.path.isfile(directory + 'dict.json'))
        self.assertEqual('a82076220e033c1ed3469d173d715df2', mls.Utils.md5_file(directory + 'dict.json'))

    def test_save_json_with_tuple_file_saves(self):
        directory = os.path.join(self.base_directory, 'logs/testing/')
        data = {'testA': [[1, 2], [3, 4]], 'testB': 'Text', 'testC': (1, 3, 5)}
        mls.FileOperation.save_dict_as_json('dict.json', directory, data)
        self.assertTrue(os.path.isfile(directory + 'dict.json'))
        self.assertEqual('bf557710f7a993fd2bf6ef547b402634', mls.Utils.md5_file(directory + 'dict.json'))

    def test_load_json_dict_loaded(self):
        directory = os.path.join(self.base_directory, '../test/files/')
        result = mls.FileOperation.load_json_as_dict('dict.json', directory)
        expected = {'testA': [[1, 2], [3, 4]], 'testB': 'Text'}
        self.assertDictEqual(expected, result)

    def test_load_json_with_tuple_dict_loaded(self):
        directory = os.path.join(self.base_directory, '../test/files/')
        result = mls.FileOperation.load_json_as_dict('dict_with_tuple.json', directory)
        expected = {'testA': [[1, 2], [3, 4]], 'testB': 'Text', 'testC': (1, 3, 5)}
        self.assertDictEqual(expected, result)

    def test_load_json_with_tuple_dict_loaded_tuple_to_string(self):
        directory = os.path.join(self.base_directory, '../test/files/')
        result = mls.FileOperation.load_json_as_dict('dict_with_tuple.json', directory, tuple_to_string=True)
        expected = {'testA': [[1, 2], [3, 4]], 'testB': 'Text', 'testC': '(1, 3, 5)'}
        self.assertDictEqual(expected, result)

    def test_save_hdf_pandas(self):
        """
        :test : mlsurvey.FileOperation.save_hdf()
        :condition : data is pandas dataframe
        :main_result : file exists
        """
        directory = os.path.join(self.base_directory, 'logs/testing/')
        data = pd.DataFrame([[1, 2], [3, 4]])
        mls.FileOperation.save_hdf('data.h5', directory, data)
        self.assertTrue(os.path.isfile(directory + 'data.h5'))
        # do not check the md5 for hdf5 files since it changes each time...

    def test_save_hdf_pandas_all_parameters(self):
        """
        :test : mlsurvey.FileOperation.save_hdf()
        :condition : data is pandas dataframe, type is 'table'
        :main_result : file exists
        """
        directory = os.path.join(self.base_directory, 'logs/testing/')
        expected_columns = ['Col1', 'Col2']
        data = pd.DataFrame([[1, 2], [3, 4]], columns=expected_columns)
        params = {'format': 'table'}
        mls.FileOperation.save_hdf('data.h5', directory, data, params)
        # File exists
        self.assertTrue(os.path.isfile(os.path.join(directory, 'data.h5')))
        # testing content
        df = mls.FileOperation.read_hdf('data.h5', directory, 'Pandas')
        np.testing.assert_array_equal(np.array([[1, 2], [3, 4]]), df.values)
        self.assertListEqual(expected_columns, df.keys().tolist())

    def test_load_hdf_pandas(self):
        """
        :test : mlsurvey.FileOperation.load_hdf()
        :condition : data is pandas dataframe, and file exists
        :main_result : dataframe is read
        """
        directory = os.path.join(self.base_directory, '../test/files/')
        expected_columns = ['Col1', 'Col2']
        df = mls.FileOperation.read_hdf('data-pandas.h5', directory, 'Pandas')
        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_array_equal(np.array([[1, 2], [3, 4]]), df.values)
        self.assertListEqual(expected_columns, df.keys().tolist())

    def test_save_json_pandas(self):
        """
        :test : mlsurvey.FileOperation.save_json()
        :condition : data is pandas dataframe
        :main_result : file exists
        """
        directory = os.path.join(self.base_directory, 'logs/testing/')
        data = pd.DataFrame([[1, 2], [3, 4]])
        mls.FileOperation.save_json('data.json', directory, data)
        self.assertTrue(os.path.isfile(os.path.join(directory, 'data.json')))

    def test_save_json_pandas_all_parameters(self):
        """
        :test : mlsurvey.FileOperation.save_json()
        :condition : data is pandas dataframe, orientation is 'index'
        :main_result : file exists and contains the correct content
        """
        directory = os.path.join(self.base_directory, 'logs/testing/')
        data = pd.DataFrame([[1, 2], [3, 4]])
        params = {'orient': 'index'}
        mls.FileOperation.save_json('data.json', directory, data, params)
        # File exists
        self.assertTrue(os.path.isfile(os.path.join(directory, 'data.json')))
        # Content is as expected
        fs = open(os.path.join(directory, 'data.json'))
        contents = fs.read()
        fs.close()
        self.assertEqual('{"0":{"0":1,"1":2},"1":{"0":3,"1":4}}', contents)

    def test_load_json_pandas(self):
        """
        :test : mlsurvey.FileOperation.load_json()
        :condition : data is pandas dataframe, and file exists
        :main_result : dataframe is read
        """
        directory = os.path.join(self.base_directory, '../test/files/')
        df = mls.FileOperation.read_json('data-pandas.json', directory, 'Pandas')
        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_array_equal(np.array([[1, 2], [3, 4]]), df.values)

    def test_save_plotly_figure(self):
        """
        :test : mlsurvey.FileOperation.save_plotly_figure()
        :condition : plotly figure is generated
        :main_result : plotly figure is saved
        """
        directory = os.path.join(self.base_directory, 'logs/testing/')
        figure = go.Figure(data=go.Bar(y=[10, 20, 30, 30]))
        mls.FileOperation.save_plotly_figure('figure-test.png', directory, figure)
        # File exists
        self.assertTrue(os.path.isfile(os.path.join(directory, 'figure-test.png')))

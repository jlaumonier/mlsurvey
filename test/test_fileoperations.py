import os
import unittest

import numpy as np
import pandas as pd

import mlsurvey as mls


class TestFileOperation(unittest.TestCase):

    def test_save_json_file_saves(self):
        directory = 'logs/testing/'
        data = {'testA': [[1, 2], [3, 4]], 'testB': 'Text'}
        mls.FileOperation.save_dict_as_json('dict.json', directory, data)
        self.assertTrue(os.path.isfile(directory + 'dict.json'))
        self.assertEqual('a82076220e033c1ed3469d173d715df2', mls.Utils.md5_file(directory + 'dict.json'))

    def test_save_json_with_tuple_file_saves(self):
        directory = 'logs/testing/'
        data = {'testA': [[1, 2], [3, 4]], 'testB': 'Text', 'testC': (1, 3, 5)}
        mls.FileOperation.save_dict_as_json('dict.json', directory, data)
        self.assertTrue(os.path.isfile(directory + 'dict.json'))
        self.assertEqual('bf557710f7a993fd2bf6ef547b402634', mls.Utils.md5_file(directory + 'dict.json'))

    def test_load_json_dict_loaded(self):
        directory = '../test/files/'
        result = mls.FileOperation.load_json_as_dict('dict.json', directory)
        expected = {'testA': [[1, 2], [3, 4]], 'testB': 'Text'}
        self.assertDictEqual(expected, result)

    def test_load_json_with_tuple_dict_loaded(self):
        directory = '../test/files/'
        result = mls.FileOperation.load_json_as_dict('dict_with_tuple.json', directory)
        expected = {'testA': [[1, 2], [3, 4]], 'testB': 'Text', 'testC': (1, 3, 5)}
        self.assertDictEqual(expected, result)

    def test_load_json_with_tuple_dict_loaded_tuple_to_string(self):
        directory = '../test/files/'
        result = mls.FileOperation.load_json_as_dict('dict_with_tuple.json', directory, tuple_to_string=True)
        expected = {'testA': [[1, 2], [3, 4]], 'testB': 'Text', 'testC': '(1, 3, 5)'}
        self.assertDictEqual(expected, result)

    def test_save_hdf_pandas(self):
        """
        :test : mlsurvey.FileOperation.save_hdf()
        :condition : data is pandas dataframe
        :main_result : file exists
        """
        directory = 'logs/testing/'
        data = pd.DataFrame([[1, 2], [3, 4]])
        mls.FileOperation.save_hdf('data.h5', directory, data)
        self.assertTrue(os.path.isfile(directory + 'data.h5'))
        # do not check the md5 for hdf5 files since it changes each time...

    def test_load_hdf_pandas(self):
        """
        :test : mlsurvey.FileOperation.load_hdf()
        :condition : data is pandas dataframe, and file exists
        :main_result : dataframe is read
        """
        directory = '../test/files/'
        df = mls.FileOperation.read_hdf('data-pandas.h5', directory, 'Pandas')
        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_array_equal(np.array([[1, 2], [3, 4]]), df.values)

    def test_save_json_pandas(self):
        """
        :test : mlsurvey.FileOperation.save_json()
        :condition : data is pandas dataframe
        :main_result : file exists
        """
        directory = 'logs/testing/'
        data = pd.DataFrame([[1, 2], [3, 4]])
        mls.FileOperation.save_json('data.json', directory, data)
        self.assertTrue(os.path.isfile(os.path.join(directory, 'data.json')))

    def test_load_json_pandas(self):
        """
        :test : mlsurvey.FileOperation.load_json()
        :condition : data is pandas dataframe, and file exists
        :main_result : dataframe is read
        """
        directory = '../test/files/'
        df = mls.FileOperation.read_json('data-pandas.json', directory, 'Pandas')
        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_array_equal(np.array([[1, 2], [3, 4]]), df.values)

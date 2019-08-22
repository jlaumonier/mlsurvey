import os
import shutil
import unittest

import mlsurvey as mls


class TestFileOperation(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('logs/testing/')

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

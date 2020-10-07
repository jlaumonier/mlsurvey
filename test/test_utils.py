import unittest

import numpy as np
import pandas as pd
import sklearn.neighbors as neighbors

import mlsurvey as mls


class TestUtils(unittest.TestCase):

    def test_md5_file(self):
        md5 = mls.Utils.md5_file('files/test_md5.txt')
        self.assertEqual('70a4b9f4707d258f559f91615297a3ec', md5)

    def test_md5_file_not_exists(self):
        """
        :test : mlsurvey.Utils.md5_file()
        :condition : unknown file
        :main_result : raise FileNotFoundError
        """
        try:
            _ = mls.Utils.md5_file('files/test_md5_unknown.txt')
            self.assertTrue(False)
        except FileNotFoundError:
            self.assertTrue(True)

    def test_import_from_dotted_path_class_created(self):
        to_import = 'sklearn.neighbors.KNeighborsClassifier'
        classdef = mls.Utils.import_from_dotted_path(to_import)
        self.assertEqual(neighbors.KNeighborsClassifier, classdef)

    def test_import_from_dotted_path_not_exists(self):
        """
        :test : mlsurvey.Utils.import_from_dotted_path()
        :condition : unknown module and class
        :main_result : AttributeError
        """
        to_import = 'sklearn.neighbors.UnknownClass'
        try:
            _ = mls.Utils.import_from_dotted_path(to_import)
            self.assertTrue(False)
        except AttributeError:
            self.assertTrue(True)

    def test_make_meshgrid(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        xx, yy = mls.Utils.make_meshgrid(x, y, h=.5)
        self.assertListEqual(xx[0].tolist(), [0, 0.5, 1, 1.5, 2, 2.5])
        self.assertListEqual(yy[0].tolist(), [2, 2, 2, 2, 2, 2])

    def test_transform_to_dict_tuple_should_transform(self):
        """
        :test : mlsurvey.Util.transform_to_dict()
        :condition : dictionary contains one __tuple__
        :main_result : transformation into tuple
        """
        base_dictionary = {'test': {'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                           'nottuple': {'t': 1},
                           'nottupleeither': 'string'}
        expected_dictionary = {'test': (1, 2, 3),
                               'nottuple': {'t': 1},
                               'nottupleeither': 'string'}
        result = mls.Utils.transform_to_dict(base_dictionary)
        self.assertDictEqual(expected_dictionary, result)

    def test_transform_to_dict_tuple_should_transform_to_string(self):
        """
        :test : mlsurvey.Util.transform_to_dict()
        :condition : dictionary contains one __tuple__
        :main_result : transforme tupple into string
        """
        base_dictionary = {'test': {'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                           'nottuple': {'t': 1},
                           'nottupleeither': 'string'}
        expected_dictionary = {'test': '(1, 2, 3)',
                               'nottuple': {'t': 1},
                               'nottupleeither': 'string'}
        result = mls.Utils.transform_to_dict(base_dictionary, tuple_to_string=True)
        self.assertDictEqual(expected_dictionary, result)

    def test_transform_to_dict_not_tuple_should_raise_error(self):
        """
        :test : mlsurvey.Util.transform_dict()
        :condition : dictionary contains one __tuple__ where __value__ is not a tuple as string
        :main_result : raise a TypeError
        """
        base_dictionary = {'test': {'__type__': '__tuple__', '__value__': '234'}}
        try:
            _ = mls.Utils.transform_to_dict(base_dictionary)
            self.assertTrue(False)
        except TypeError:
            self.assertTrue(True)

    def test_transform_to_dict_tuple_nested_should_transform(self):
        """
        :test : mlsurvey.Util.transform_to_dict()
        :condition : dictionary contains __tuple__ with nested dictionaries
        :main_result : transformation into tuples and lists
        """
        base_dictionary = {'test': {'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                           'nottuple': {'a': {'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                                        'b': {'c': 1,
                                              'd': {'__type__': '__tuple__', '__value__': '(1, 2, 3)'}}},
                           'nottupleeither': 'string'}
        expected_dictionary = {'test': (1, 2, 3),
                               'nottuple': {'a': (1, 2, 3),
                                            'b': {'c': 1,
                                                  'd': (1, 2, 3)}},
                               'nottupleeither': 'string'}
        result = mls.Utils.transform_to_dict(base_dictionary)
        self.assertDictEqual(expected_dictionary, result)

    def test_transform_to_dict_list_of_tuple_nested_should_transform(self):
        """
        :test : mlsurvey.Utils.transform_to_dict()
        :condition : dictionary contains lists of __tuple__ with nested dictionaries
        :main_result : transformation into tuples
        """
        base_dictionary = {'test': [{'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                                    {'__type__': '__tuple__', '__value__': '(4, 5, 6)'}],
                           'nottuple': {'a': {'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                                        'b': {'c': 1,
                                              'd': [{'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                                                    {'__type__': '__tuple__', '__value__': '(4, 5, 6)'}]}},
                           'nottupleeither': 'string'}
        expected_dictionary = {'test': [(1, 2, 3), (4, 5, 6)],
                               'nottuple': {'a': (1, 2, 3),
                                            'b': {'c': 1,
                                                  'd': [(1, 2, 3), (4, 5, 6)]}},
                               'nottupleeither': 'string'}
        result = mls.Utils.transform_to_dict(base_dictionary)
        self.assertDictEqual(expected_dictionary, result)

    def test_transform_to_json_tuple_should_transform(self):
        """
        :test : mlsurvey.Util.transform_to_json()
        :condition : dictionary contains one tuple
        :main_result : transformation into __tuple__
        """
        base_dictionary = {'test': (1, 2, 3),
                           'nottuple': {'t': 1},
                           'nottupleeither': 'string'}
        expected_dictionary = {'test': {'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                               'nottuple': {'t': 1},
                               'nottupleeither': 'string'}
        result = mls.Utils.transform_to_json(base_dictionary)
        self.assertDictEqual(expected_dictionary, result)

    def test_transform_to_json_tuple_nested_should_transform(self):
        """
        :test : mlsurvey.Util.transform_to_json()
        :condition : dictionary contains __tuple__ with nested dictionaries
        :main_result : transformation into tuples and lists
        """
        base_dictionary = {'test': (1, 2, 3),
                           'nottuple': {'a': (1, 2, 3),
                                        'b': {'c': 1,
                                              'd': (1, 2, 3)}},
                           'nottupleeither': 'string'}
        expected_dictionary = {'test': {'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                               'nottuple': {'a': {'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                                            'b': {'c': 1,
                                                  'd': {'__type__': '__tuple__', '__value__': '(1, 2, 3)'}}},
                               'nottupleeither': 'string'}
        result = mls.Utils.transform_to_json(base_dictionary)
        self.assertDictEqual(expected_dictionary, result)

    def test_transform_to_json_list_of_tuple_nested_should_transform(self):
        """
        :test : mlsurvey.Utils.transform_to_json()
        :condition : dictionary contains lists of __tuple__ with nested dictionaries
        :main_result : transformation into tuples and lists
        """
        base_dictionary = {'test': [(1, 2, 3), (4, 5, 6)],
                           'nottuple': {'a': (1, 2, 3),
                                        'b': {'c': 1,
                                              'd': [(1, 2, 3), (4, 5, 6)]}},
                           'nottupleeither': 'string'}
        expected_dictionary = {'test': [{'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                                        {'__type__': '__tuple__', '__value__': '(4, 5, 6)'}],
                               'nottuple': {'a': {'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                                            'b': {'c': 1,
                                                  'd': [{'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                                                        {'__type__': '__tuple__', '__value__': '(4, 5, 6)'}]}},
                               'nottupleeither': 'string'}
        result = mls.Utils.transform_to_json(base_dictionary)
        self.assertDictEqual(expected_dictionary, result)

    def test_check_dict_python_ready_should_be_ready(self):
        """
        :test : mlsurvey.Utils.check_dict_python_ready()
        :condition : dictionary does not contains '__type__': '__tuple__'
        :main result : dictionary is python-ready
        """
        base_dictionary = {'test': [(1, 2, 3), (4, 5, 6)],
                           'nottuple': {'a': (1, 2, 3),
                                        'b': {'c': 1,
                                              'd': [[1, 2, 3], [4, 5, 6]]}},
                           'nottupleeither': 'string'}
        result = mls.Utils.check_dict_python_ready(base_dictionary)
        self.assertTrue(result)

    def test_check_dict_python_ready_should_not_be_ready(self):
        """
        :test : mlsurvey.Utils.check_dict_python_ready()
        :condition : dictionary does contains '__type__': '__tuple__'
        :main result : dictionary is not python-ready
        """
        base_dictionary = {'test': [{'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                                    {'__type__': '__tuple__', '__value__': '(4, 5, 6)'}],
                           'nottuple': {'a': {'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                                        'b': {'c': 1}},
                           'nottupleeither': 'string'}
        result = mls.Utils.check_dict_python_ready(base_dictionary)
        self.assertFalse(result)

    def test_check_dict_python_ready_should_not_be_ready_with_list_only(self):
        """
        :test : mlsurvey.Utils.check_dict_python_ready()
        :condition : dictionary does contains '__type__': '__tuple__' into list
        :main result : dictionary is not python-ready
        """
        base_dictionary = {'test': [{'__type__': '__tuple__', '__value__': '(1, 2, 3)'},
                                    {'__type__': '__tuple__', '__value__': '(4, 5, 6)'}],
                           'nottupleeither': 'string'}
        result = mls.Utils.check_dict_python_ready(base_dictionary)
        self.assertFalse(result)

    def test_is_dataframe_empty_pandas_is_empty(self):
        """
        :test : mlsurvey.Utils.is_dataframe_empty()
        :condition : dataframe is pandas and empty
        :main_result : True
        """
        df = pd.DataFrame()
        result = mls.Utils.is_dataframe_empty(df)
        self.assertTrue(result)

    def test_is_dataframe_empty_pandas_is_not_empty(self):
        """
        :test : mlsurvey.Utils.is_dataframe_empty()
        :condition : dataframe is pandas and not empty
        :main_result : False
        """
        x = np.array([[1, 2], [3, 4]])
        df = pd.DataFrame(data=x)
        result = mls.Utils.is_dataframe_empty(df)
        self.assertFalse(result)

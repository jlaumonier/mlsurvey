import unittest
import os
import mlsurvey as mls


class TestExpandDict(unittest.TestCase):
    config_directory = ''
    base_directory = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.base_directory = os.path.join(directory, '.')
        cls.config_directory = os.path.join(cls.base_directory, 'config/')

    def test_dict_generator_cartesian_product(self):
        source = {"element1": ["A", "B", "C"], "element2": [1, 2], "element3": [True, False], "element4": 12}
        result = list(mls.ExpandDict.dict_generator_cartesian_product(source))
        self.assertEqual(12, len(result))
        expected_first_dict = {"element1": "A", "element2": 1, "element3": True, "element4": 12}
        self.assertDictEqual(expected_first_dict, result[0])
        expected_second_dict = {"element1": "A", "element2": 1, "element3": False, "element4": 12}
        self.assertDictEqual(expected_second_dict, result[1])

    def test_dict_generator_cartesian_product_string_not_list(self):
        source = {"element1": "test"}
        result = list(mls.ExpandDict.dict_generator_cartesian_product(source))
        self.assertEqual(1, len(result))
        expected_first_dict = {"element1": "test"}
        self.assertDictEqual(expected_first_dict, result[0])

    def test_dict_generator_cartesian_product_none(self):
        """
       :test : mlsurvey.Utils.dict_generator_cartesian_product()
       :condition : source contains None
       :main_result : result contains None also
       """
        source = {"element1": None}
        result = list(mls.ExpandDict.dict_generator_cartesian_product(source))
        self.assertEqual(1, len(result))
        expected_first_dict = {"element1": None}
        self.assertDictEqual(expected_first_dict, result[0])

    def test_dict_generator_cartesian_product_empty(self):
        source = {}
        result = list(mls.ExpandDict.dict_generator_cartesian_product(source))
        self.assertEqual(1, len(result))
        expected_first_dict = {}
        self.assertDictEqual(expected_first_dict, result[0])

    def test_run_no_need_expand(self):
        input_dict = {'key1': 'value1'}
        expected_result = [{'key1': 'value1'}]
        result = mls.ExpandDict.run(input_dict)
        self.assertListEqual(expected_result, result)

    def test_run_no_need_expand_two_keys(self):
        input_dict = {'key1': 'value1', 'key2': 'value2'}
        expected_result = [{'key1': 'value1', 'key2': 'value2'}]
        result = mls.ExpandDict.run(input_dict)
        self.assertListEqual(expected_result, result)

    def test_run_simple_expand(self):
        input_dict = {'key1': ['value1', 'value2']}
        expected_result = [{'key1': 'value1'}, {'key1': 'value2'}]
        result = mls.ExpandDict.run(input_dict)
        self.assertListEqual(expected_result, result)

    def test_run_simple_expand_with_one_fixed(self):
        input_dict = {'key1': ['value1', 'value2'], 'key2': 'value3'}
        expected_result = [{'key1': 'value1', 'key2': 'value3'}, {'key1': 'value2', 'key2': 'value3'}]
        result = mls.ExpandDict.run(input_dict)
        self.assertListEqual(expected_result, result)

    def test_run_list_inside_list(self):
        input_dict = {'key3': [{'key1': ['value1', 'value2']}, {'key1': 'value3'}]}
        expected_result = [{'key3': {'key1': 'value1'}}, {'key3': {'key1': 'value2'}}, {'key3': {'key1': 'value3'}}]
        result = mls.ExpandDict.run(input_dict)
        self.assertListEqual(expected_result, result)

    def test_run_two_keys_expand(self):
        input_dict = {'key1': ['value1', 'value2'], 'key2': ['value3', 'value4']}
        expected_result = [{'key1': 'value1', 'key2': 'value3'},
                           {'key1': 'value1', 'key2': 'value4'},
                           {'key1': 'value2', 'key2': 'value3'},
                           {'key1': 'value2', 'key2': 'value4'}]
        result = mls.ExpandDict.run(input_dict)
        self.assertListEqual(expected_result, result)

    def test_run_one_keys_inside_expand(self):
        input_dict = {'key3': {'key1': ['value1', 'value2']}}
        expected_result = [{'key3': {'key1': 'value1'}},
                           {'key3': {'key1': 'value2'}}]
        result = mls.ExpandDict.run(input_dict)
        self.assertListEqual(expected_result, result)

    def test_run_two_keys_inside_expand(self):
        input_dict = {'key3': {'key1': ['value1', 'value2']},
                      'key4': {'key5': ['value1', 'value2']}}
        expected_result = [{'key3': {'key1': 'value1'}, 'key4': {'key5': 'value1'}},
                           {'key3': {'key1': 'value1'}, 'key4': {'key5': 'value2'}},
                           {'key3': {'key1': 'value2'}, 'key4': {'key5': 'value1'}},
                           {'key3': {'key1': 'value2'}, 'key4': {'key5': 'value2'}}]
        result = mls.ExpandDict.run(input_dict)
        self.assertListEqual(expected_result, result)

    def test_run_complex_expand(self):
        input_dict = {'key3': {'key1': ['value1', 'value2'],
                               'key2': ['value3', 'value4'],
                               'key4': 'value5'},
                      'key5': 'value6',
                      'key6': {'key7': {'key8': ['value7', 'value8']}}}
        expected_result_0 = {'key3': {'key1': 'value1',
                                      'key2': 'value3',
                                      'key4': 'value5'},
                             'key5': 'value6',
                             'key6': {'key7': {'key8': 'value7'}}}

        result = mls.ExpandDict.run(input_dict)
        self.assertEqual(8, len(result))
        self.assertDictEqual(expected_result_0, result[0])

    def test_expand_likely_config(self):
        input_dict = {
            'learning_process': {
                'parameters': {'algorithm': {'algorithm-family': 'sklearn.neighbors.KNeighborsClassifier',
                                             'hyperparameters': {'algorithm': 'auto',
                                                                 'n_neighbors': [2, 3],
                                                                 'weights': ['uniform', 'auto']}},
                               'input': {'parameters': {'n_samples': 100,
                                                        'noise': 0,
                                                        'random_state': 0,
                                                        'shuffle': True},
                                         'type': 'NClassRandomClassificationWithNoise'},
                               'split': [{'parameters': {'random_state': 0,
                                                         'shuffle': True,
                                                         'test_size': 20},
                                          'type': 'traintest'},
                                         {'parameters': {'random_state': 0,
                                                         'shuffle': True,
                                                         'test_size': 40},
                                          'type': 'traintest'}
                                         ]}
            }
        }
        expected_result_1 = {
            'learning_process': {
                'parameters': {'algorithm': {'algorithm-family': 'sklearn.neighbors.KNeighborsClassifier',
                                             'hyperparameters': {'algorithm': 'auto',
                                                                 'n_neighbors': 2,
                                                                 'weights': 'uniform'}},
                               'input': {'parameters': {'n_samples': 100,
                                                        'noise': 0,
                                                        'random_state': 0,
                                                        'shuffle': True},
                                         'type': 'NClassRandomClassificationWithNoise'},
                               'split': {'parameters': {'random_state': 0,
                                                        'shuffle': True,
                                                        'test_size': 40},
                                         'type': 'traintest'},
                               }
            }
        }
        result = mls.ExpandDict.run(input_dict)
        self.assertEqual(8, len(result))
        self.assertDictEqual(expected_result_1, result[1])

import unittest
import os

import mlsurvey as mls


class TestConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.base_directory = os.path.join(directory, '')

    def test_load_config_config_loaded(self):
        config = mls.Config('config_loaded.json', directory=os.path.join(self.base_directory, 'config'))
        self.assertEqual('config loaded', config.data['testconfig'])
        self.assertTrue(mls.Utils.check_dict_python_ready(config.data))

    def test_load_config_default_config_loaded(self):
        config = mls.Config(directory=os.path.join(self.base_directory, 'config'))
        self.assertEqual('config loaded', config.data['testconfig'])
        self.assertTrue(mls.Utils.check_dict_python_ready(config.data))

    def test_get_dataset_dataset_config_obtained(self):
        config = mls.Config('complete_config_loaded.json', directory=os.path.join(self.base_directory, 'config'))
        self.assertEqual('NClassRandomClassificationWithNoise', config.data['#refs']['datasets']['DataSet1']['type'])
        self.assertEqual(100, config.data['#refs']['datasets']['DataSet1']['parameters']['n_samples'])
        self.assertTrue(mls.Utils.check_dict_python_ready(config.data))

    def test_load_config_from_other_directory(self):
        config = mls.Config('config_loaded.json', directory=os.path.join(self.base_directory, 'files'))
        self.assertEqual('config loaded', config.data['testconfig'])
        self.assertTrue(mls.Utils.check_dict_python_ready(config.data))

    def test_load_config_from_other_directory_without_end_slash(self):
        config = mls.Config('config_loaded.json', directory=os.path.join(self.base_directory, 'files'))
        self.assertEqual('config loaded', config.data['testconfig'])
        self.assertTrue(mls.Utils.check_dict_python_ready(config.data))

    def test_load_multiple_config_config_loaded(self):
        config = mls.Config('multiple_config.json', directory=os.path.join(self.base_directory, 'config'))
        self.assertEqual('NClassRandomClassificationWithNoise', config.data['#refs']['datasets']['DataSet1']['type'])
        self.assertListEqual(['@datasets.DataSet1', '@datasets.DataSet2', '@datasets.DataSet3'],
                             config.data['learning_process']['parameters']['input'])
        self.assertTrue(mls.Utils.check_dict_python_ready(config.data))

    def test_load_config_file_not_python_ready_config_loaded(self):
        config = mls.Config('full_multiple_config.json', directory=os.path.join(self.base_directory, 'config'))
        self.assertTrue(mls.Utils.check_dict_python_ready(config.data))

    def test_init_config_with_dictionary(self):
        c = {'testconfig': 'config loaded'}
        config = mls.Config(config=c)
        self.assertEqual('config loaded', config.data['testconfig'])
        self.assertTrue(mls.Utils.check_dict_python_ready(config.data))

    def test_init_config_with_dictionary_not_python_ready(self):
        c = {'testconfig': {"__type__": "__tuple__", "__value__": "(1, 2, 3)"}}
        config = mls.Config(config=c)
        self.assertTupleEqual((1, 2, 3), config.data['testconfig'])
        self.assertTrue(mls.Utils.check_dict_python_ready(config.data))

    def test_init_config_file_not_exists(self):
        """
        :test : mlsurvey.Config()
        :condition : Config file not exist
        :main_result : raise FileNotFoundError
        """
        try:
            _ = mls.Config('config_loaded_not_exists.json', directory=os.path.join(self.base_directory, 'config'))
            self.assertTrue(False)
        except FileNotFoundError:
            self.assertTrue(True)

    def test_init_config_file_not_json(self):
        """
        :test : mlsurvey.Config()
        :condition : config file is not a json file
        :main_result : raise ConfigError
        """
        try:
            _ = mls.Config('config_loaded_not_json.json', directory=os.path.join(self.base_directory, 'config'))
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_init_config_application_config_loaded(self):
        """
        :test : mlsurvey.Config()
        :condition : app_config.json exists
        :main_result : application config loaded
        """
        config = mls.Config('config.json', directory=os.path.join(self.base_directory, 'config'))
        self.assertFalse(config.app_config['app_section']['value'])

    def test_compact_should_compact(self):
        """
        :test : mlsurvey.Config.compact()
        :condition : config file format to compacted config format
        :main_result : transformation ok
        """
        base_config = {
            '#refs': {'algorithms': {'knn-base': {'type': 'sklearn.neighbors.KNeighborsClassifier',
                                                  'hyperparameters': {'algorithm': 'auto',
                                                                      'n_neighbors': 2,
                                                                      'weights': 'uniform'}}
                                     },
                      'datasets': {'DataSetNClassRandom': {'parameters': {'n_samples': [100, 200],
                                                                          'noise': 0,
                                                                          'random_state': 0,
                                                                          'shuffle': True},
                                                           'type': 'NClassRandomClassificationWithNoise'}},
                      'splits': {'traintest20': {'parameters': {'random_state': 0,
                                                                'shuffle': True,
                                                                'test_size': 20},
                                                 'type': 'traintest'},
                                 'traintest40': {'parameters': {'random_state': 0,
                                                                'shuffle': True,
                                                                'test_size': 40},
                                                 'type': 'traintest'}
                                 }},
            'learning_process': {'parameters': {'algorithm': '@algorithms.knn-base',
                                                'input': '@datasets.DataSetNClassRandom',
                                                'split': ['@splits.traintest20', '@splits.traintest40']},
                                 },
            }
        expected_config = {
            'learning_process': {
                'parameters': {'algorithm': {'type': 'sklearn.neighbors.KNeighborsClassifier',
                                             'hyperparameters': {'algorithm': 'auto',
                                                                 'n_neighbors': 2,
                                                                 'weights': 'uniform'}},
                               'input': {'parameters': {'n_samples': [100, 200],
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
        config = mls.Config(config=base_config)
        config.compact()
        self.assertDictEqual(expected_config, config.data)
        self.assertTrue(config.is_compacted())

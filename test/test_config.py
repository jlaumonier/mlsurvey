import unittest

import mlsurvey as mls


class TestConfig(unittest.TestCase):

    def test_load_config_config_loaded(self):
        config = mls.Config('config_loaded.json')
        self.assertEqual('config loaded', config.data['testconfig'])

    def test_load_config_default_config_loaded(self):
        config = mls.Config()
        self.assertEqual('config loaded', config.data['testconfig'])

    def test_get_dataset_dataset_config_obtained(self):
        config = mls.Config('complete_config_loaded.json')
        self.assertEqual('NClassRandomClassificationWithNoise', config.data['datasets']['DataSet1']['type'])
        self.assertEqual(100, config.data['datasets']['DataSet1']['parameters']['n_samples'])

    def test_load_config_from_other_directory(self):
        config = mls.Config('config_loaded.json', 'files/')
        self.assertEqual('config loaded', config.data['testconfig'])

    def test_load_config_from_other_directory_without_end_slash(self):
        config = mls.Config('config_loaded.json', 'files')
        self.assertEqual('config loaded', config.data['testconfig'])

    def test_load_multiple_config_config_loaded(self):
        config = mls.Config('multiple_config.json')
        self.assertEqual('NClassRandomClassificationWithNoise', config.data['datasets']['DataSet1']['type'])
        self.assertListEqual(['DataSet1', 'DataSet2', 'DataSet3'], config.data['learning_process']['input'])

    def test_init_config_with_dictionary(self):
        c = {'testconfig': 'config loaded'}
        config = mls.Config(config=c)
        self.assertEqual('config loaded', config.data['testconfig'])

    def test_init_config_file_not_exists(self):
        """
        :test : mlsurvey.Config()
        :condition : Config file not exist
        :main_result : raise FileNotFoundError
        """
        try:
            _ = mls.Config('config_loaded_not_exists.json')
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
            _ = mls.Config('config_loaded_not_json.json')
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_compactt_should_compact(self):
        """
        :test : mlsurvey.Config.transform()
        :condition : config file format to compacted config format
        :main_result : transformation ok
        """
        base_config = {'algorithms': {'knn-base': {'algorithm-family': 'sklearn.neighbors.KNeighborsClassifier',
                                                   'hyperparameters': {'algorithm': 'auto',
                                                                       'n_neighbors': 2,
                                                                       'weights': 'uniform'}}},
                       'datasets': {'DataSetNClassRandom': {'parameters': {'n_samples': 100,
                                                                           'noise': 0,
                                                                           'random_state': 0,
                                                                           'shuffle': True},
                                                            'type': 'NClassRandomClassificationWithNoise'}},
                       'learning_process': {'algorithm': 'knn-base',
                                            'input': 'DataSetNClassRandom',
                                            'split': 'traintest20'},
                       'splits': {'traintest20': {'parameters': {'random_state': 0,
                                                                 'shuffle': True,
                                                                 'test_size': 20},
                                                  'type': 'traintest'}}}
        expected_config = {
            'learning_process': {'algorithm': {'algorithm-family': 'sklearn.neighbors.KNeighborsClassifier',
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
                                                          'test_size': 20},
                                           'type': 'traintest'}}
        }
        actual_config_result = mls.Config.compact(base_config)
        self.assertDictEqual(expected_config, actual_config_result)

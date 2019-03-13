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
        self.assertEqual('NClassRandomClassification', config.data['datasets']['DataSet1']['type'])
        self.assertEqual(100, config.data['datasets']['DataSet1']['parameters']['n_samples'])

    def test_load_config_from_other_directory(self):
        config = mls.Config('config_loaded.json', 'files/')
        self.assertEqual('config loaded', config.data['testconfig'])

    def test_load_multiple_config_config_loaded(self):
        config = mls.Config('multiple_config.json')
        self.assertEqual('NClassRandomClassification', config.data['datasets']['DataSet1']['type'])
        self.assertListEqual(['DataSet1', 'DataSet2', 'DataSet3'], config.data['learning_process']['input'])

    def test_init_config_with_dictionary(self):
        c = {'testconfig': 'config loaded'}
        config = mls.Config(config=c)
        self.assertEqual('config loaded', config.data['testconfig'])

import os
import unittest

import tinydb as tdb

import mlsurvey as mls


class TestAnalyzeLogs(unittest.TestCase):
    directory = ''

    @classmethod
    def setUpClass(cls):
        d = os.path.dirname(__file__)
        cls.directory = os.path.join(d, '../files/visualize-log//')

    def test_init_should_init(self):
        analyse_logs = mls.visualize.AnalyzeLogs(self.directory)
        expected_list_dir = ['directory1', 'directory2', 'directory3']
        expected_list_full_dir = [os.path.join(self.directory, 'directory1'),
                                  os.path.join(self.directory, 'directory2'),
                                  os.path.join(self.directory, 'directory3')]
        self.assertEqual(self.directory, analyse_logs.directory)
        self.assertEqual(len(analyse_logs.list_dir), len(analyse_logs.list_full_dir))
        self.assertListEqual(analyse_logs.list_dir, expected_list_dir)
        self.assertListEqual(analyse_logs.list_full_dir, expected_list_full_dir)
        self.assertIsInstance(analyse_logs.db, tdb.database.TinyDB)
        self.assertEqual(0, len(analyse_logs.db.all()))

    def test_store_config_should_fill_database(self):
        """
        :test : mlsurvey.visualize.Analyse_logs.store_config()
        :condition : config files present in self.directory
        :main_result : all config files are inserted into the database
        """
        analyse_logs = mls.visualize.AnalyzeLogs(self.directory)
        analyse_logs.store_config()
        q = tdb.Query()
        r = analyse_logs.db.search(
            q.learning_process.algorithm['algorithm-family'] == 'sklearn.neighbors.KNeighborsClassifier')
        expected_result = {
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
                                           'type': 'traintest'}},
            'location': os.path.join(self.directory, 'directory1')
        }
        self.assertEqual(3, len(analyse_logs.db.all()))
        self.assertListEqual([expected_result], r)

    def test_store_config_should_answer_query1(self):
        """
        :test : mlsurvey.visualize.Analyse_logs.store_config()
        :condition : config files present in self.directory
        :main_result : answer learning_process.input.type == 'NClassRandomClassificationWithNoise'
        """
        analyse_logs = mls.visualize.AnalyzeLogs(self.directory)
        analyse_logs.store_config()
        q = tdb.Query()
        r = analyse_logs.db.search(
            q.learning_process.input.type == 'NClassRandomClassificationWithNoise')
        self.assertEqual(3, len(r))

    def test_store_config_should_answer_query2(self):
        """
        :test : mlsurvey.visualize.Analyse_logs.store_config()
        :condition : config files present in self.directory
        :main_result : answer (learning_process.input.parameters.n_samples == 100)
                              | (learning_process.input.parameters.n_samples == 10000)
        """
        analyse_logs = mls.visualize.AnalyzeLogs(self.directory)
        analyse_logs.store_config()
        q = tdb.Query()
        r = analyse_logs.db.search(
            (q.learning_process.input.parameters.n_samples == 100)
            | (q.learning_process.input.parameters.n_samples == 10000))
        self.assertEqual(3, len(r))

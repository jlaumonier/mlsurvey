import os
import unittest

import tinydb as tdb

import mlsurvey as mls


class TestAnalyzeLogs(unittest.TestCase):
    directory = ''

    @classmethod
    def setUpClass(cls):
        d = os.path.dirname(__file__)
        cls.directory = os.path.join(d, '../files/visualize-log/')
        cls.directory_images = os.path.join(d, '../files/analyse-log-images/')

    def test_init_should_init(self):
        analyse_logs = mls.visualize.AnalyzeLogs(self.directory)
        expected_list_full_dir = [os.path.join(self.directory, 'directory1'),
                                  os.path.join(self.directory, 'directory2'),
                                  os.path.join(self.directory, 'directory3'),
                                  os.path.join(self.directory, 'directory4')]
        self.assertEqual(self.directory, analyse_logs.directory)
        self.assertListEqual(expected_list_full_dir, analyse_logs.list_full_dir)
        self.assertIsInstance(analyse_logs.lists, dict)
        self.assertIsNone(analyse_logs.parameters_df)
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
            q.learning_process.parameters.algorithm['type'] == 'sklearn.neighbors.KNeighborsClassifier')
        expected_result = {
            'learning_process': {
                'type': 'mlsurvey.sl.workflows.SupervisedLearningWorkflow',
                'parameters': {'algorithm': {'type': 'sklearn.neighbors.KNeighborsClassifier',
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
                                         'type': 'traintest'}}},
            'location': os.path.join(self.directory, 'directory1')
        }
        self.assertEqual(4, len(analyse_logs.db.all()))
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
            q.learning_process.parameters.input.type == 'NClassRandomClassificationWithNoise')
        self.assertEqual(4, len(r))

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
            (q.learning_process.parameters.input.parameters.n_samples == 100)
            | (q.learning_process.parameters.input.parameters.n_samples == 10000))
        self.assertEqual(4, len(r))

    def test_store_config_should_fill_list(self):
        """
        :test : mlsurvey.visualize.Analyse_logs.store_config()
        :condition : config files present in self.directory
        :main_result : fill_lists has been called
        """
        analyse_logs = mls.visualize.AnalyzeLogs(self.directory)
        analyse_logs.store_config()
        self.assertEqual(4, len(analyse_logs.lists['algorithm']))
        self.assertEqual(2, len(analyse_logs.lists['input']))
        self.assertEqual(4, len(analyse_logs.parameters_df))

    def test_fill_list_should_fill_images_and_html(self):
        """
        :test : mlsurvey.visualize.Analyse_logs.fill_list_images()
        :condition : config files present in self.directory
        :main_result : images and html files (recursively) lists are returned
        """
        expected_image_files_lists = ['dir1/image1.png', 'dir1/image2.png',
                                      'dir2/image3.jpg', 'dir2/image4.png',
                                      'image5.png']
        expected_json_files_lists = ['algorithm.json',
                                     'config.json',
                                     'data-content.json',
                                     'data.json',
                                     'dataset.json',
                                     'evaluation.json',
                                     'raw_data-content.json',
                                     'raw_data.json',
                                     'raw_test-content.json',
                                     'raw_train-content.json',
                                     'split_data.json',
                                     'terminated.json',
                                     'test-content.json',
                                     'train-content.json']
        analyse_logs = mls.visualize.AnalyzeLogs(self.directory_images)
        analyse_logs.store_config()
        self.assertListEqual(expected_image_files_lists, analyse_logs.lists['image_files'])
        self.assertListEqual(expected_json_files_lists, analyse_logs.lists['json_files'])

    def test_fill_lists_should_fill(self):
        """
        :test : mlsurvey.visualize.Analyse_logs.fill_lists()
        :condition : config files present in self.directory
        :main_result : algorithms_list and datasets_list filled with possibles choices
        """
        doc1 = {'learning_process': {'parameters': {'algorithm': {'type': 'Algorithm 1'},
                                                    'input': {'type': 'Dataset 1'},
                                                    'split': {'type': 'traintest'}}}}
        doc2 = {'learning_process': {'parameters': {'algorithm': {'type': 'Algorithm 2'},
                                                    'input': {'type': 'Dataset 1'},
                                                    'split': {'type': 'traintest'}}}}
        doc3 = {'learning_process': {'parameters': {'algorithm': {'type': 'Algorithm 2'},
                                                    'input': {'type': 'Dataset 1'},
                                                    'split': {'type': 'traintest'}}}}
        expected_algorithms_list = ['.', 'Algorithm 1', 'Algorithm 2']
        expected_datasets_list = ['.', 'Dataset 1']
        expected_paramaters_df = [['Algorithm 1', 'Dataset 1', 'traintest'],
                                  ['Algorithm 2', 'Dataset 1', 'traintest'],
                                  ['Algorithm 2', 'Dataset 1', 'traintest']]
        expected_image_files_lists = []
        analyse_logs = mls.visualize.AnalyzeLogs(self.directory)
        analyse_logs.db.insert(doc1)
        analyse_logs.db.insert(doc2)
        analyse_logs.db.insert(doc3)
        analyse_logs.fill_lists()
        self.assertListEqual(expected_algorithms_list, analyse_logs.lists['algorithm'])
        self.assertListEqual(expected_datasets_list, analyse_logs.lists['input'])
        self.assertListEqual(expected_paramaters_df, analyse_logs.parameters_df.values.tolist())
        self.assertListEqual(expected_image_files_lists, analyse_logs.lists['image_files'])

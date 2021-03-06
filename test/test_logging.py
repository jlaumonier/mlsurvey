import os
import shutil
import unittest
import mlflow.tracking

import plotly.graph_objects as go

from sklearn import neighbors

import mlsurvey as mls


class TestLogging(unittest.TestCase):
    base_directory = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.base_directory = os.path.join(directory, '')
        cls.mlflow_client = mlflow.tracking.MlflowClient()
        cls.mlflow_experiments = cls.mlflow_client.list_experiments()

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir, ignore_errors=True)

    def test_init_log_directory_created_with_date(self):
        """
        :test : mlsurvey.Logging()
        :condition : -
        :main_result : no error. everything is created as wanted
        """
        log = mls.Logging()
        # dir_name is initialized
        self.assertIsNotNone(log.dir_name)
        # directory is set as wanted
        self.assertEqual(os.path.join(log.base_dir, log.dir_name), log.directory)
        # log directory not created
        self.assertFalse(os.path.isdir(log.directory))
        # mlflow initialization
        self.assertIsInstance(log.mlflow_client, mlflow.tracking.MlflowClient)
        self.assertIsNone(log.mlflow_run_id)

    def test_init_log_directory_mlflow_run_id_set(self):
        """
        :test : mlsurvey.Logging()
        :condition : set mlflow_run_id
        :main_result : the log.mlflow_run_id is set
        """
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        log = mls.Logging(mlflow_run_id=run.info.run_id)
        self.assertIsInstance(log.mlflow_run_id, str)

    def test_init_log_directory_create_with_fixed_name(self):
        dir_name = 'testing/'
        log = mls.Logging(dir_name=dir_name)
        self.assertEqual(os.path.join('logs/', dir_name), log.directory)

    def test_init_log_directory_dir_exists(self):
        """
        :test : mlsurvey.Logging()
        :condition : Directory already exists
        :main_result : no error
        """
        base_dir = 'files/'
        dir_name = 'slw'
        self.assertTrue(os.path.isdir(os.path.join(self.base_directory, base_dir, dir_name)))
        _ = mls.Logging(base_dir=base_dir, dir_name=dir_name)
        self.assertTrue(True)

    def test_save_inputs_input_saved(self):
        """
        :test : mlsurvey.Logging.save_input()
        :condition : data initialized
        :main_result : data and metadata are saved
        """
        dir_name = 'testing/'
        d_data = mls.sl.datasets.DataSetFactory.create_dataset('make_moons')
        d_data.params['random_state'] = 0
        i_data = mls.sl.models.DataFactory.create_data('Pandas', d_data.generate())
        d_train = mls.sl.datasets.DataSetFactory.create_dataset('load_iris')
        i_train = mls.sl.models.DataFactory.create_data('Pandas', d_train.generate())
        d_test = mls.sl.datasets.DataSetFactory.create_dataset('make_circles')
        d_test.params['random_state'] = 0
        i_test = mls.sl.models.DataFactory.create_data('Pandas', d_test.generate())
        log = mls.Logging(dir_name)
        inputs = {'data': i_data, 'train': i_train, 'test': i_test}
        log.save_input(inputs)
        self.assertTrue(os.path.isfile(log.directory + 'data-content.h5'))
        self.assertTrue(os.path.isfile(log.directory + 'train-content.h5'))
        self.assertTrue(os.path.isfile(log.directory + 'test-content.h5'))
        self.assertTrue(os.path.isfile(log.directory + 'data-content.json'))
        self.assertTrue(os.path.isfile(log.directory + 'train-content.json'))
        self.assertTrue(os.path.isfile(log.directory + 'test-content.json'))
        self.assertTrue(os.path.isfile(log.directory + 'input.json'))
        self.assertEqual('397a9f41ea039cbaf5bc4a8fb78c9b23', mls.Utils.md5_file(log.directory + 'input.json'))
        try:
            _ = log.mlflow_client.get_run(log.mlflow_run_id)
            self.assertTrue(False)
        except TypeError:
            self.assertTrue(True)

    def test_save_inputs_input_saved_and_mlflow_artifacts(self):
        """
        :test : mlsurvey.Logging.save_input()
        :condition : data initialized, mlflow run is initialized
        :main_result : artifacts are saved
        """
        dir_name = 'testing/'
        d_data = mls.sl.datasets.DataSetFactory.create_dataset('make_moons')
        d_data.params['random_state'] = 0
        i_data = mls.sl.models.DataFactory.create_data('Pandas', d_data.generate())
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        log = mls.Logging(dir_name, mlflow_run_id=run.info.run_id)
        inputs = {'data': i_data}
        log.save_input(inputs)
        list_artifact = [i.path for i in log.mlflow_client.list_artifacts(log.mlflow_run_id)]
        self.assertIn('data-content.json', list_artifact)
        self.assertIn('input.json', list_artifact)
        self.assertEqual(2, len(list_artifact))

    def test_save_inputs_other_name_input_saved(self):
        """
        :test : mlsurvey.Logging.save_input()
        :condition : filename is set
        :main_result : input are saved
        """
        dir_name = 'testing/'
        d_data = mls.sl.datasets.DataSetFactory.create_dataset('make_moons')
        d_data.params['random_state'] = 0
        i_data = mls.sl.models.DataFactory.create_data('Pandas', d_data.generate())
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        log = mls.Logging(dir_name, mlflow_run_id=run.info.run_id)
        inputs = {'data': i_data}
        log.save_input(inputs, metadata_filename='test.json')
        self.assertTrue(os.path.isfile(log.directory + 'test.json'))
        list_artifact = [i.path for i in log.mlflow_client.list_artifacts(log.mlflow_run_id)]
        self.assertIn('test.json', list_artifact)

    def test_save_inputs_inputs_none_input_saved(self):
        dir_name = 'testing-none-input/'
        log = mls.Logging(dir_name)
        inputs = {'data': None, 'train': None, 'test': None}
        log.save_input(inputs)
        self.assertFalse(os.path.isfile(log.directory + 'data-content.h5'))
        self.assertFalse(os.path.isfile(log.directory + 'train-content.h5'))
        self.assertFalse(os.path.isfile(log.directory + 'test-content.h5'))
        self.assertTrue(os.path.isfile(log.directory + 'input.json'))
        self.assertEqual('c6e977bcc44c3435cf59b9cced4538e0', mls.Utils.md5_file(log.directory + 'input.json'))

    def test_load_input_input_loaded(self):
        dir_name = 'files/input_load/'
        log = mls.Logging(dir_name, base_dir=os.path.join(self.base_directory, '../test/'))
        results = log.load_input('input.json')
        i = results['test']
        self.assertEqual(0.6459514595757855, i.x[0, 0])
        self.assertEqual(1.0499271427368027, i.x[0, 1])
        self.assertEqual(1.0, i.y[0])
        self.assertEqual(1, i.y_pred[0])
        self.assertEqual(2, i.x.shape[1])
        self.assertEqual(20, i.x.shape[0])
        self.assertEqual(20, i.y.shape[0])
        self.assertEqual(20, i.y_pred.shape[0])

    def test_save_json_file_saves(self):
        """
        :test : mlsurvey.Logging.save_json()
        :condition : filename is set
        :main_result : dictionary is saved as json, and artifacts are in mlflow trakcing
        """
        dir_name = 'testing/'
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        log = mls.Logging(dir_name, mlflow_run_id=run.info.run_id)
        d = {'testA': [[1, 2], [3, 4]], 'testB': 'Text'}
        log.save_dict_as_json('dict.json', d)
        # dir exists
        self.assertTrue(os.path.isdir(log.base_dir + log.dir_name + '/'))
        # file exists
        self.assertTrue(os.path.isfile(log.directory + 'dict.json'))
        # file content is ok
        self.assertEqual('a82076220e033c1ed3469d173d715df2', mls.Utils.md5_file(log.directory + 'dict.json'))
        list_artifact = [i.path for i in log.mlflow_client.list_artifacts(log.mlflow_run_id)]
        # file is in the mlflow artifacts
        self.assertIn('dict.json', list_artifact)
        # there can be only one
        self.assertEqual(1, len(list_artifact))

    def test_load_json_dict_loaded(self):
        dir_name = 'files/'
        log = mls.Logging(dir_name, base_dir=os.path.join(self.base_directory, '../test/'))
        result = log.load_json_as_dict('dict.json')
        expected = {'testA': [[1, 2], [3, 4]], 'testB': 'Text'}
        self.assertDictEqual(expected, result)

    def test_save_classifier(self):
        dir_name = 'testing/'
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        log = mls.Logging(dir_name, mlflow_run_id=run.info.run_id)
        classifier = neighbors.KNeighborsClassifier()
        log.save_classifier(classifier)
        # classifier exists
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'model.joblib')))
        classifier = log.load_classifier()
        # classifier is of the correct type
        self.assertIsInstance(classifier, neighbors.KNeighborsClassifier)
        # classifier content is ok
        self.assertEqual(30, classifier.get_params()['leaf_size'])
        list_artifact = [i.path for i in log.mlflow_client.list_artifacts(log.mlflow_run_id)]
        # file is in the mlflow artifacts
        self.assertIn('model.joblib', list_artifact)
        # there can be only one
        self.assertEqual(1, len(list_artifact))

    def test_save_classifier_filename_provided(self):
        dir_name = 'testing/'
        log = mls.Logging(dir_name)
        classifier = neighbors.KNeighborsClassifier()
        log.save_classifier(classifier, filename='test_model.joblib')
        self.assertTrue(os.path.isfile(os.path.join(log.directory + 'test_model.joblib')))
        classifier = log.load_classifier(filename='test_model.joblib')
        self.assertIsInstance(classifier, neighbors.KNeighborsClassifier)
        self.assertEqual(30, classifier.get_params()['leaf_size'])

    def test_load_classifier(self):
        dir_name = 'files/slw'
        log = mls.Logging(dir_name, base_dir=os.path.join(self.base_directory, '../test/'))
        classifier = log.load_classifier()
        self.assertIsInstance(classifier, neighbors.KNeighborsClassifier)

    def test_load_classifier_filename_provided(self):
        dir_name = 'files/slw'
        log = mls.Logging(dir_name, base_dir=os.path.join(self.base_directory, '../test/'))
        classifier = log.load_classifier(filename='test_model.joblib')
        self.assertIsInstance(classifier, neighbors.KNeighborsClassifier)

    def test_log_config_config_logged(self):
        """
        :test : mlsurvey.Logging.log_config()
        :condition : config is a dict
        :main_result : config is logged into json and mlflow
        """
        dict_config = {
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
                               'split': {'parameters': {'random_state': 0,
                                                        'shuffle': True,
                                                        'test_size': 20},
                                         'type': 'traintest'}
                               }
            }
        }
        dir_name = 'testing/'
        filename = 'config.json'
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        log = mls.Logging(dir_name, mlflow_run_id=run.info.run_id)
        log.log_config(filename, dict_config)
        # file exists
        self.assertTrue(os.path.isfile(os.path.join(log.directory, filename)))
        # logged into mlflow
        self.assertIn('input.type', log.mlflow_client.get_run(log.mlflow_run_id).data.params)

    def test_log_metrics_metrics_logged(self):
        """
        :test : mlsurvey.Logging.log_metrics()
        :condition : config is a dict
        :main_result : config is logged into json and mlflow
        """
        dict_metrics = {"type": "EvaluationSupervised", "score": 0.85, "confusion_matrix": [[3, 2], [1, 14]],
                        "sub_evaluation": {"type": "EvaluationFairness",
                                           "demographic_parity": -0.11142691040630448,
                                           "equal_opportunity": 0.033609877188398224,
                                           "statistical_parity": -0.08789806860021965,
                                           "average_equalized_odds": -0.03792762377938433,
                                           "disparate_impact_rate": 0.8535902223731116}}
        dir_name = 'testing/'
        filename = 'config.json'
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        log = mls.Logging(dir_name, mlflow_run_id=run.info.run_id)
        log.log_metrics(filename, dict_metrics)
        # file exists
        self.assertTrue(os.path.isfile(os.path.join(log.directory, filename)))
        # logged into mlflow
        self.assertIn('sub_evaluation.demographic_parity', log.mlflow_client.get_run(log.mlflow_run_id).data.metrics)

    def test_save_plotly_figure(self):
        """
        :test : mlsurvey.Logging.log_plotly_figure()
        :condition : filename is set
        :main_result : figure is saved as image
        """
        dir_name = 'testing/'
        sub_dir = 'plot/'
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        log = mls.Logging(dir_name, mlflow_run_id=run.info.run_id)
        figure1 = go.Figure(data=go.Bar(y=[10, 20, 30, 30]))
        figure2 = go.Figure(data=go.Bar(y=[10, 20, 30, 30]))
        dict_fig = {'figure1.png': figure1, 'figure2.png': figure2}
        log.save_plotly_figures(dict_fig, sub_dir)
        # dirs exists
        self.assertTrue(os.path.isdir(os.path.join(log.base_dir, log.dir_name, sub_dir)))
        # file exists
        self.assertTrue(os.path.isfile(os.path.join(log.directory, sub_dir, 'figure1.png')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, sub_dir, 'figure2.png')))
        list_artifact = [i.path for i in log.mlflow_client.list_artifacts(log.mlflow_run_id)]
        # file is in the mlflow artifacts
        self.assertIn('figure1.png', list_artifact)
        # there can be only one
        self.assertEqual(2, len(list_artifact))

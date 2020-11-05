import os
import shutil
import unittest

import luigi
from sklearn import neighbors
import mlflow

import mlsurvey as mls


class TestLearnTask(unittest.TestCase):
    config_directory = ''
    base_directory = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.base_directory = os.path.join(directory, '../../../')
        cls.config_directory = os.path.join(cls.base_directory, 'config/')
        cls.mlflow_client = mlflow.tracking.MlflowClient()
        cls.mlflow_experiments = cls.mlflow_client.list_experiments()

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(os.path.join(cls.base_directory, log.base_dir), ignore_errors=True)

    def test_run(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LearnTask.run()
        :condition : data file are split, saved in hdf database and logged
        :main_result : model is trained
        """
        temp_log = mls.Logging()
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        luigi.build([mls.sl.workflows.tasks.LearnTask(logging_directory=temp_log.dir_name,
                                                      logging_base_directory=os.path.join(self.base_directory,
                                                                                          temp_log.base_dir),
                                                      config_filename='complete_config_loaded.json',
                                                      config_directory=self.config_directory,
                                                      mlflow_run_id=run.info.run_id)],
                    local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'model.joblib')))
        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'algorithm.json')))
        classifier = log.load_classifier()
        self.assertIsInstance(classifier, neighbors.KNeighborsClassifier)
        self.assertEqual(30, classifier.get_params()['leaf_size'])

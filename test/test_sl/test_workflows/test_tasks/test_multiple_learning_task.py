import os
import shutil
import unittest

import luigi
import mlflow

import mlsurvey as mls


class TestMultipleLearningTask(unittest.TestCase):
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
        :test : mlsurvey.sl.workflows.tasks.TestMultipleLearningTask.run()
        :condition : config file contains multiple learning config
        :main_result : all learning have ran
        """
        temp_log = mls.Logging()
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        luigi.build([mls.sl.workflows.tasks.MultipleLearningTask(logging_directory=temp_log.dir_name,
                                                                 logging_base_directory=os.path.join(
                                                                     self.base_directory,
                                                                     temp_log.base_dir),
                                                                 config_filename='multiple_config.json',
                                                                 config_directory=self.config_directory,
                                                                 base_directory=self.base_directory,
                                                                 mlflow_run_id=run.info.run_id)],
                    local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'results.json')))
        result_dict = log.load_json_as_dict('results.json')
        self.assertEqual(3, result_dict['NbLearning'])

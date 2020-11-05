import os
import shutil
import unittest

import mlflow
import luigi
import numpy as np

import mlsurvey as mls


class TestEvaluateTask(unittest.TestCase):
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
        :test : mlsurvey.sl.workflows.tasks.EvaluateTask.run()
        :condition : model is trained, config is made to learn on dataset without fairness config
        :main_result : model is evaluated, fairness is not calculated
        """
        temp_log = mls.Logging()
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        luigi.build([mls.sl.workflows.tasks.EvaluateTask(logging_directory=temp_log.dir_name,
                                                         logging_base_directory=os.path.join(self.base_directory,
                                                                                             temp_log.base_dir),
                                                         config_filename='complete_config_loaded.json',
                                                         config_directory=self.config_directory,
                                                         base_directory=self.base_directory,
                                                         mlflow_run_id=run.info.run_id)],
                    local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir),
                          dir_name=temp_log.dir_name,
                          mlflow_run_id=run.info.run_id)
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'evaluation.json')))
        evaluation_dict = log.load_json_as_dict('evaluation.json')
        evaluation = mls.sl.models.EvaluationFactory.create_instance_from_dict(evaluation_dict)

        expected_cm = np.array([[13, 0], [1, 6]])
        self.assertEqual(0.95, evaluation.score)
        np.testing.assert_array_equal(expected_cm, evaluation.confusion_matrix)
        self.assertIsNone(evaluation.sub_evaluation)
        # metric in mlflow
        self.assertIn('score', log.mlflow_client.get_run(log.mlflow_run_id).data.metrics)

    def test_run_with_fairness_evaluated(self):
        """
        :test : mlsurvey.sl.workflows.tasks.EvaluateTask.run()
        :condition : model is trained
        :main_result : model is evaluated with fairness metrics
        """
        temp_log = mls.Logging()
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        luigi.build([mls.sl.workflows.tasks.EvaluateTask(logging_directory=temp_log.dir_name,
                                                         logging_base_directory=os.path.join(self.base_directory,
                                                                                             temp_log.base_dir),
                                                         config_filename='config_filedataset.json',
                                                         config_directory=self.config_directory,
                                                         base_directory=self.base_directory,
                                                         mlflow_run_id=run.info.run_id)],
                    local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir),
                          dir_name=temp_log.dir_name,
                          mlflow_run_id=run.info.run_id)
        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'evaluation.json')))
        evaluation_dict = log.load_json_as_dict('evaluation.json')
        evaluation = mls.sl.models.EvaluationFactory.create_instance_from_dict(evaluation_dict)

        self.assertIsNotNone(evaluation.sub_evaluation)
        self.assertAlmostEqual(-0.3666666, evaluation.sub_evaluation.demographic_parity, delta=1e-07)
        self.assertAlmostEqual(0.476190476, evaluation.sub_evaluation.disparate_impact_rate, delta=1e-07)
        self.assertAlmostEqual(1.0, evaluation.sub_evaluation.equal_opportunity, delta=1e-07)
        self.assertAlmostEqual(-0.8, evaluation.sub_evaluation.statistical_parity, delta=1e-07)
        self.assertAlmostEqual(-0.6666666, evaluation.sub_evaluation.average_equalized_odds, delta=1e-07)
        # fairness in mlflow
        self.assertIn('sub_evaluation.disparate_impact_rate', log.mlflow_client.get_run(log.mlflow_run_id).data.metrics)

    def test_run_with_fairness_evaluated_with_real_data(self):
        """
        :test : mlsurvey.sl.workflows.tasks.EvaluateTask.run()
        :condition : model is trained with real data
        :main_result : model is evaluated with fairness metrics
        """
        temp_log = mls.Logging()
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        luigi.build([mls.sl.workflows.tasks.EvaluateTask(logging_directory=temp_log.dir_name,
                                                         logging_base_directory=os.path.join(self.base_directory,
                                                                                             temp_log.base_dir),
                                                         config_filename='config_filedataset_real_data.json',
                                                         config_directory=self.config_directory,
                                                         base_directory=self.base_directory,
                                                         mlflow_run_id=run.info.run_id)],
                    local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'evaluation.json')))
        evaluation_dict = log.load_json_as_dict('evaluation.json')
        evaluation = mls.sl.models.EvaluationFactory.create_instance_from_dict(evaluation_dict)

        self.assertIsNotNone(evaluation.sub_evaluation)
        self.assertIsNotNone(evaluation.sub_evaluation.demographic_parity)

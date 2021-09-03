import os
import shutil

import mlflow

from kedro.pipeline.node import Node
from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner

import numpy as np

import mlsurvey as mls


class TestEvaluateTask(mls.testing.TaskTestCase):
    config_directory = ''
    base_directory = ''
    mlflow_client = None
    mlflow_experiments = None

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

    def test_get_node(self):
        """
        :test : mlsurvey.sl.workflows.tasks.EvaluateTask.get_node()
        :condition : -
        :main_result : create a kedro with input and output parameter
        """
        evaluate_node = mls.sl.workflows.tasks.EvaluateTask.get_node()
        self.assertIsInstance(evaluate_node, Node)

    def test_evaluate(self):
        """
        :test : mlsurvey.sl.workflows.tasks.EvaluateTask.evaluate()
        :condition : model is trained, config is made to learn on dataset without fairness config
        :main_result : model is evaluated, fairness is not calculated
        """
        config, log = self._init_config_log('complete_config_loaded.json',
                                            self.base_directory,
                                            self.config_directory)

        dataset_dict = mls.FileOperation.load_json_as_dict('dataset.json',
                                                           os.path.join(self.base_directory, 'files/tasks/load_data'))
        dataset = mls.sl.datasets.DataSetFactory.create_dataset_from_dict(dataset_dict)

        df_raw_data = mls.FileOperation.read_hdf('raw_data-content.h5',
                                                 os.path.join(self.base_directory, 'files/tasks/load_data'),
                                                 'Pandas')
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data)

        df_prepared_data = mls.FileOperation.read_hdf('data-content.h5',
                                                      os.path.join(self.base_directory, 'files/tasks/prepare_data'),
                                                      'Pandas')
        prepared_data = mls.sl.models.DataFactory.create_data('Pandas', df_prepared_data)

        df_test_data = mls.FileOperation.read_hdf('test-content.h5',
                                                  os.path.join(self.base_directory, 'files/tasks/split_data'),
                                                  'Pandas')
        test_data = mls.sl.models.DataFactory.create_data('Pandas', df_test_data)

        model_fullpath = os.path.join(self.base_directory, 'files/tasks/learn', 'model.joblib')

        [evaluation] = mls.sl.workflows.tasks.EvaluateTask.evaluate(config, log, dataset,
                                                                    raw_data, prepared_data,
                                                                    test_data, model_fullpath)

        expected_cm = np.array([[13, 0], [1, 6]])
        self.assertEqual(0.95, evaluation.score)
        np.testing.assert_array_equal(expected_cm, evaluation.confusion_matrix)
        self.assertIsNone(evaluation.sub_evaluation)

    def _run_one_task(self, config_filename):
        # create node from Task
        load_data_node = mls.workflows.tasks.LoadDataTask.get_node()
        prepare_data_node = mls.sl.workflows.tasks.PrepareDataTask.get_node()
        split_data_node = mls.sl.workflows.tasks.SplitDataTask.get_node()
        learn_node = mls.sl.workflows.tasks.LearnTask.get_node()
        evaluate_node = mls.sl.workflows.tasks.EvaluateTask.get_node()
        config, log = self._init_config_log(config_filename, self.base_directory, self.config_directory)
        # Prepare a data catalog
        data_catalog = DataCatalog({'config': MemoryDataSet(),
                                    'log': MemoryDataSet(),
                                    'base_directory': MemoryDataSet()})
        data_catalog.save('config', config)
        data_catalog.save('log', log)
        data_catalog.save('base_directory', self.base_directory)
        # Assemble nodes into a pipeline
        pipeline = Pipeline([load_data_node, prepare_data_node, split_data_node, learn_node, evaluate_node])
        # Create a runner to run the pipeline
        runner = SequentialRunner()
        # Run the pipeline
        runner.run(pipeline, data_catalog)
        return log, config, data_catalog

    def test_run(self):
        """
        :test : mlsurvey.sl.workflows.tasks.EvaluateTask.evaluate()
        :condition : model is trained, config is made to learn on dataset without fairness config
        :main_result : model is evaluated, fairness is not calculated
        """
        config_filename = 'complete_config_loaded.json'
        log, config, data_catalog = self._run_one_task(config_filename)
        log.set_sub_dir(str(mls.sl.workflows.tasks.EvaluateTask.__name__))
        # evaluation file exists
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'evaluation.json')))
        # metric in mlflow
        self.assertIn('score', log.mlflow_client.get_run(log.mlflow_current_run.info.run_id).data.metrics)

    def test_run_with_fairness_evaluated(self):
        """
        :test : mlsurvey.sl.workflows.tasks.EvaluateTask.run()
        :condition : model is trained
        :main_result : model is evaluated with fairness metrics
        """
        config_filename = 'config_filedataset.json'
        log, config, data_catalog = self._run_one_task(config_filename)
        log.set_sub_dir(str(mls.sl.workflows.tasks.EvaluateTask.__name__))

        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'evaluation.json')))
        evaluation_dict = log.load_json_as_dict('evaluation.json')
        evaluation = mls.sl.models.EvaluationFactory.create_instance_from_dict(evaluation_dict)

        self.assertIsNotNone(evaluation.sub_evaluation)
        self.assertAlmostEqual(-0.3666666, evaluation.sub_evaluation.demographic_parity, delta=1e-07)
        self.assertAlmostEqual(0.476190476, evaluation.sub_evaluation.disparate_impact_rate, delta=1e-07)
        self.assertAlmostEqual(1.0, evaluation.sub_evaluation.equal_opportunity, delta=1e-07)
        self.assertAlmostEqual(-0.8, evaluation.sub_evaluation.statistical_parity, delta=1e-07)
        self.assertAlmostEqual(-0.6666666, evaluation.sub_evaluation.average_equalized_odds, delta=1e-07)
        # fairness in mlflow
        self.assertIn('sub_evaluation.disparate_impact_rate',
                      log.mlflow_client.get_run(log.mlflow_current_run.info.run_id).data.metrics)

    def test_run_with_fairness_evaluated_with_real_data(self):
        """
        :test : mlsurvey.sl.workflows.tasks.EvaluateTask.run()
        :condition : model is trained with real data
        :main_result : model is evaluated with fairness metrics
        """
        config_filename = 'config_filedataset_real_data.json'
        log, config, data_catalog = self._run_one_task(config_filename)
        log.set_sub_dir(str(mls.sl.workflows.tasks.EvaluateTask.__name__))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'evaluation.json')))
        evaluation_dict = log.load_json_as_dict('evaluation.json')
        evaluation = mls.sl.models.EvaluationFactory.create_instance_from_dict(evaluation_dict)

        self.assertIsNotNone(evaluation.sub_evaluation)
        self.assertIsNotNone(evaluation.sub_evaluation.demographic_parity)

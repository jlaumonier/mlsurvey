import os
import shutil
import unittest

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler

import mlsurvey as mls


class TestSupervisedLearningWorkflow(unittest.TestCase):
    cd = ''
    bd = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.cd = os.path.join(directory, '../../config/')
        cls.bd = os.path.join(directory, '../../')

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir, ignore_errors=True)

    def test_init_SL_workflow_should_initialized(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        self.assertIsNotNone(slw.config.data)
        self.assertEqual(self.cd, slw.config_directory)
        self.assertFalse(slw.terminated)
        self.assertFalse(slw.task_terminated_get_data)
        self.assertFalse(slw.task_terminated_prepare_data)
        self.assertFalse(slw.task_terminated_split_data)
        self.assertFalse(slw.task_terminated_learn)
        self.assertFalse(slw.task_terminated_fairness)
        self.assertFalse(slw.task_terminated_evaluate)
        self.assertFalse(slw.task_terminated_persistence)
        self.assertIsInstance(slw.data_preparation, StandardScaler)
        self.assertIsInstance(slw.context, mls.sl.models.Context)
        self.assertIsInstance(slw.log, mls.Logging)

    def test_init_config_file_not_exists_should_be_not_terminated(self):
        """
        :test : mlsurvey.workflows.SupervisedLearningWorkflow()
        :condition : config file not exists
        :main_result : init is not terminated
        """
        slw = mls.sl.workflows.SupervisedLearningWorkflow('multiple_config_not_exist.json', config_directory=self.cd)
        self.assertIsInstance(slw, mls.sl.workflows.SupervisedLearningWorkflow)
        self.assertFalse(slw.task_terminated_init)

    def test_init_config_file_not_a_json_file_should_stop(self):
        """
        :test : mlsurvey.workflows.SupervisedLearningWorkflow()
        :condition : config file is not a json file
        :main_result : init is not terminated
        """
        slw = mls.sl.workflows.SupervisedLearningWorkflow('config_loaded_not_json.json', config_directory=self.cd)
        self.assertIsInstance(slw, mls.sl.workflows.SupervisedLearningWorkflow)
        self.assertFalse(slw.task_terminated_init)

    def test_init_SL_workflow_should_initialized_with_config(self):
        c = {'testconfig': 'config loaded'}
        slw = mls.sl.workflows.SupervisedLearningWorkflow(config=c)
        self.assertEqual('config loaded', slw.config.data['testconfig'])

    def test_init_SL_workflow_logging_in_specified_dir(self):
        """
        :test : mlsurvey.workflows.SupervisedLearningWorkflow()
        :condition : set the logging directory
        :main_result : logging directory is set as specified
        """
        expected_dir = 'testlog/'
        slw = mls.sl.workflows.SupervisedLearningWorkflow(logging_dir=expected_dir)
        self.assertEqual('logs/' + expected_dir, slw.log.directory)

    def test_set_terminated_all_terminated(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_fairness = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertTrue(slw.terminated)

    def test_set_terminated_all_terminated_but_init(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = False
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_fairness = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_get_data(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = False
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_fairness = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_prepare_data(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = False
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_fairness = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_split_data(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = False
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_fairness = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_learn(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = False
        slw.task_terminated_evaluate = True
        slw.task_terminated_fairness = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_evaluate(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = False
        slw.task_terminated_fairness = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_fairness(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_fairness = False
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_persistence(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_fairness = True
        slw.task_terminated_persistence = False
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_task_get_data_data_is_obtained(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow('complete_config_loaded.json', config_directory=self.cd)
        slw.task_get_data()
        self.assertIsInstance(slw.context.dataset, mls.sl.datasets.NClassRandomClassificationWithNoise)
        self.assertDictEqual({}, slw.context.dataset.fairness)
        self.assertIsNotNone(slw.context.raw_data)
        self.assertEqual(100, len(slw.context.raw_data.x))
        self.assertEqual(100, len(slw.context.raw_data.y))
        self.assertIsNone(slw.context.data)
        self.assertTrue(slw.task_terminated_get_data)

    def test_task_get_data_filedataset_data_is_obtained_pandas_implicit(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow('config_filedataset.json',
                                                          config_directory=self.cd,
                                                          base_directory=self.bd)
        slw.task_get_data()
        self.assertIsInstance(slw.context.dataset, mls.sl.datasets.FileDataSet)
        self.assertDictEqual({'protected_attribute': 1, 'privileged_classes': 'x >= 25'}, slw.context.dataset.fairness)
        self.assertIsNotNone(slw.context.raw_data)
        self.assertIsInstance(slw.context.raw_data.df, pd.DataFrame)
        self.assertEqual(13, len(slw.context.raw_data.x))
        self.assertEqual(13, len(slw.context.raw_data.y))
        self.assertIsNone(slw.context.data)
        self.assertTrue(slw.task_terminated_get_data)

    def test_task_get_data_filedataset_data_is_obtained_pandas_explicit(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow('config_filedataset.json',
                                                          config_directory=self.cd,
                                                          base_directory=self.bd)
        slw.task_get_data()
        self.assertIsInstance(slw.context.dataset, mls.sl.datasets.FileDataSet)
        self.assertDictEqual({'protected_attribute': 1, 'privileged_classes': 'x >= 25'}, slw.context.dataset.fairness)
        self.assertIsNotNone(slw.context.raw_data)
        self.assertIsInstance(slw.context.raw_data.df, pd.DataFrame)
        self.assertEqual(13, len(slw.context.raw_data.x))
        self.assertEqual(13, len(slw.context.raw_data.y))
        self.assertIsNone(slw.context.data)
        self.assertTrue(slw.task_terminated_get_data)

    def test_task_get_data_filedataset_data_is_obtained_csv_dask_metadata_explicit(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow('config_filedataset_csv_dask.json',
                                                          config_directory=self.cd,
                                                          base_directory=self.bd)
        slw.task_get_data()
        self.assertIsInstance(slw.context.dataset, mls.sl.datasets.FileDataSet)
        self.assertIsNotNone(slw.context.raw_data)
        self.assertIsInstance(slw.context.raw_data.df, dd.DataFrame)
        self.assertEqual('rating', slw.context.raw_data.y_col_name)
        self.assertEqual(39, len(slw.context.raw_data.x))
        self.assertEqual(39, len(slw.context.raw_data.y))
        self.assertIsNone(slw.context.data)
        self.assertTrue(slw.task_terminated_get_data)

    def test_task_prepare_data_data_should_be_prepared(self):
        """
        :test : mlsurvey.workflows.SupervisedLearningWorkflow.prepare_data()
        :condition : data is generated as pandas dataframe (should work with dask)
        :main_result : data is transformed with StandardScaler()
        """
        slw = mls.sl.workflows.SupervisedLearningWorkflow('complete_config_loaded.json', config_directory=self.cd)
        slw.task_get_data()
        lx = len(slw.context.raw_data.x)
        ly = len(slw.context.raw_data.y)
        self.assertEqual(-1.766054694735782, slw.context.raw_data.x[0][0])
        slw.task_prepare_data()
        self.assertEqual(-0.7655005998158294, slw.context.data.x[0][0])
        self.assertEqual(lx, len(slw.context.data.x))
        self.assertEqual(ly, len(slw.context.data.y))
        self.assertTrue(slw.task_terminated_prepare_data)

    def test_task_split_data_data_should_be_split_train_test_pandas(self):
        """
        :test : mlsurvey.workflows.SupervisedLearningWorkflow.task_split_data()
        :condition : data loaded as pandas. Should work with dask
        :main_result : data is split
        """
        slw = mls.sl.workflows.SupervisedLearningWorkflow('complete_config_loaded.json', config_directory=self.cd)
        slw.task_get_data()
        slw.task_prepare_data()
        self.assertEqual(100, len(slw.context.data.x))
        slw.task_split_data()
        self.assertEqual(100, len(slw.context.data.x))
        self.assertEqual(100, len(slw.context.data.y))
        self.assertEqual(20, len(slw.context.data_test.x))
        self.assertEqual(20, len(slw.context.data_test.y))
        self.assertEqual(80, len(slw.context.data_train.x))
        self.assertEqual(80, len(slw.context.data_train.y))
        self.assertTrue(slw.task_terminated_split_data)

    def test_task_learn_classifier_should_have_learn(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow('complete_config_loaded.json', config_directory=self.cd)
        slw.task_get_data()
        slw.task_prepare_data()
        slw.task_split_data()
        slw.task_learn()
        self.assertIsInstance(slw.context.classifier, neighbors.KNeighborsClassifier)
        # ADD some assert to test the learning
        self.assertTrue(slw.task_terminated_learn)

    def test_task_evaluate_classifier_should_have_evaluate_pandas(self):
        """
        :test : mlsurvey.workflows.SupervisedLearningWorkflow.task_evaluate()
        :condition : data loaded as pandas
        :main_result : the evaluation is completed
        """
        slw = mls.sl.workflows.SupervisedLearningWorkflow('complete_config_loaded.json', config_directory=self.cd)
        slw.task_get_data()
        slw.task_prepare_data()
        slw.task_split_data()
        slw.task_learn()
        slw.task_evaluate()
        expected_cm = np.array([[13, 0], [1, 6]])
        self.assertEqual(0.95, slw.context.evaluation.score)
        np.testing.assert_array_equal(expected_cm, slw.context.evaluation.confusion_matrix)
        self.assertEqual(100, len(slw.context.raw_data.y_pred))
        self.assertTrue(not np.isnan(slw.context.raw_data.y_pred).all())
        self.assertEqual(100, len(slw.context.data.y_pred))
        self.assertTrue(not np.isnan(slw.context.data.y_pred).all())
        self.assertEqual(80, len(slw.context.data_train.y_pred))
        self.assertTrue(not np.isnan(slw.context.data_train.y_pred).all())
        self.assertEqual(20, len(slw.context.data_test.y_pred))
        self.assertTrue(not np.isnan(slw.context.data_test.y_pred).all())
        self.assertIsNone(slw.context.evaluation.sub_evaluation)
        self.assertTrue(slw.task_terminated_evaluate)

    def test_task_evaluate_classifier_should_have_evaluate_dask(self):
        """
        :test : mlsurvey.workflows.SupervisedLearningWorkflow.task_evaluate()
        :condition : data loaded as dask
        :main_result : the evaluation is completed
        """
        slw = mls.sl.workflows.SupervisedLearningWorkflow('complete_config_loaded_dask.json', config_directory=self.cd)
        slw.task_get_data()
        slw.task_prepare_data()
        slw.task_split_data()
        slw.task_learn()
        slw.task_evaluate()
        expected_cm = np.array([[13, 0], [1, 6]])
        self.assertEqual(0.95, slw.context.evaluation.score)
        np.testing.assert_array_equal(expected_cm, slw.context.evaluation.confusion_matrix)
        self.assertEqual(100, len(slw.context.raw_data.y_pred))
        self.assertTrue(not np.isnan(slw.context.raw_data.y_pred).all())
        self.assertEqual(100, len(slw.context.data.y_pred))
        self.assertTrue(not np.isnan(slw.context.data.y_pred).all())
        self.assertEqual(80, len(slw.context.data_train.y_pred))
        self.assertTrue(not np.isnan(slw.context.data_train.y_pred).all())
        self.assertEqual(20, len(slw.context.data_test.y_pred))
        self.assertTrue(not np.isnan(slw.context.data_test.y_pred).all())
        self.assertIsNone(slw.context.evaluation.sub_evaluation)
        self.assertTrue(slw.task_terminated_evaluate)

    def test_task_evaluate_classifier_should_have_evaluate_dask_csv(self):
        """
        :test : mlsurvey.workflows.SupervisedLearningWorkflow.task_evaluate()
        :condition : csv data loaded as dask
        :main_result : the evaluation is terminated
        """
        slw = mls.sl.workflows.SupervisedLearningWorkflow('config_filedataset_csv_dask.json',
                                                          config_directory=self.cd,
                                                          base_directory=self.bd)
        slw.task_get_data()
        slw.task_prepare_data()
        slw.task_split_data()
        slw.task_learn()
        slw.task_evaluate()
        self.assertTrue(slw.task_terminated_evaluate)

    def test_task_fairness_classifier_should_have_calculate_fairness(self):
        """
        :test : mlsurvey.workflows.SupervisedLearningWorkflow.fairness()
        :condition : config is made to learn on dataset with fairness config
        :main_result : fairness is calculate
        """
        slw = mls.sl.workflows.SupervisedLearningWorkflow('config_filedataset.json',
                                                          config_directory=self.cd,
                                                          base_directory=self.bd)
        slw.task_get_data()
        slw.task_prepare_data()
        slw.task_split_data()
        slw.task_learn()
        slw.task_evaluate()
        slw.task_fairness()
        self.assertIsInstance(slw.context.evaluation.sub_evaluation, mls.sl.models.EvaluationFairness)
        self.assertAlmostEqual(-0.3666666, slw.context.evaluation.sub_evaluation.demographic_parity, delta=1e-07)
        self.assertAlmostEqual(1.0, slw.context.evaluation.sub_evaluation.equal_opportunity, delta=1e-07)
        self.assertAlmostEqual(-0.8, slw.context.evaluation.sub_evaluation.statistical_parity, delta=1e-07)
        self.assertAlmostEqual(-0.6666666, slw.context.evaluation.sub_evaluation.average_equalized_odds, delta=1e-07)
        self.assertAlmostEqual(0.476190476, slw.context.evaluation.sub_evaluation.disparate_impact_rate, delta=1e-07)
        self.assertTrue(slw.task_terminated_fairness)

    def test_task_fairness_classifier_not_executed(self):
        """
        :test : mlsurvey.workflows.SupervisedLearningWorkflow.fairness()
        :condition : config is made to learn on dataset without fairness config
        :main_result : fairness is not calculated
        """
        slw = mls.sl.workflows.SupervisedLearningWorkflow('complete_config_loaded.json',
                                                          config_directory=self.cd)
        slw.task_get_data()
        slw.task_prepare_data()
        slw.task_split_data()
        slw.task_learn()
        slw.task_evaluate()
        slw.task_fairness()
        self.assertIsNone(slw.context.evaluation.sub_evaluation)
        self.assertTrue(slw.task_terminated_fairness)

    def test_task_persist_data_classifier_should_have_been_saved(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow('complete_config_loaded.json', config_directory=self.cd)
        slw.task_get_data()
        slw.task_prepare_data()
        slw.task_split_data()
        slw.task_learn()
        slw.task_evaluate()
        slw.task_persist()
        self.assertTrue(os.path.isfile(slw.log.directory + 'config.json'))
        self.assertEqual('ae28ce16852d7a5ddbecdb07ea755339', mls.Utils.md5_file(slw.log.directory + 'config.json'))
        self.assertTrue(os.path.isfile(slw.log.directory + 'dataset.json'))
        self.assertEqual('8c61e35b282706649f63fba48d3d103e', mls.Utils.md5_file(slw.log.directory + 'dataset.json'))
        self.assertTrue(os.path.isfile(slw.log.directory + 'input.json'))
        self.assertEqual('7eea77c934e04107af044ab15398ac16', mls.Utils.md5_file(slw.log.directory + 'input.json'))
        self.assertTrue(os.path.isfile(slw.log.directory + 'algorithm.json'))
        self.assertEqual('1697475bd77100f5a9c8806c462cbd0b', mls.Utils.md5_file(slw.log.directory + 'algorithm.json'))
        self.assertTrue(os.path.isfile(slw.log.directory + 'model.joblib'))
        self.assertEqual('cdd31d2c409b9910fc9a6a92e5cabaef', mls.Utils.md5_file(slw.log.directory + 'model.joblib'))
        self.assertTrue(os.path.isfile(slw.log.directory + 'evaluation.json'))
        self.assertEqual('3880646a29148f80a36efd2eb14e8814', mls.Utils.md5_file(slw.log.directory + 'evaluation.json'))
        self.assertTrue(slw.task_terminated_persistence)

    def test_run_all_step_should_be_executed(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow('complete_config_loaded.json', config_directory=self.cd)
        self.assertFalse(slw.terminated)
        slw.run()

        # data is generated
        self.assertTrue(slw.task_terminated_get_data)
        # data is prepared
        self.assertTrue(slw.task_terminated_prepare_data)
        # data is split
        self.assertTrue(slw.task_terminated_split_data)
        # learning in done
        self.assertTrue(slw.task_terminated_learn)
        # evaluation in done
        self.assertTrue(slw.task_terminated_evaluate)
        # fairness is done
        self.assertTrue(slw.task_terminated_fairness)
        # persistence in done
        self.assertTrue(slw.task_terminated_persistence)

        # all tasks are finished
        self.assertTrue(slw.terminated)

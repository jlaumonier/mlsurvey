import os
import shutil
import unittest

from sklearn import neighbors
from sklearn.preprocessing import StandardScaler

import mlsurvey as mls


class TestSupervisedLearningWorkflow(unittest.TestCase):
    cd = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.cd = os.path.join(directory, '../config/')

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir)

    def test_init_SL_workflow_should_initialized(self):
        slw = mls.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        self.assertIsNotNone(slw.config.data)
        self.assertFalse(slw.terminated)
        self.assertFalse(slw.task_terminated_get_data)
        self.assertFalse(slw.task_terminated_prepare_data)
        self.assertFalse(slw.task_terminated_split_data)
        self.assertFalse(slw.task_terminated_learn)
        self.assertFalse(slw.task_terminated_evaluate)
        self.assertFalse(slw.task_terminated_persistence)
        self.assertIsInstance(slw.data_preparation, StandardScaler)
        self.assertIsInstance(slw.context, mls.models.Context)
        self.assertIsInstance(slw.log, mls.Logging)

    def test_init_config_file_not_exists_should_be_not_terminated(self):
        """
        :test : mlsurvey.workflows.SupervisedLearningWorkflow()
        :condition : config file not exists
        :main_result : init is not terminated
        """
        slw = mls.workflows.SupervisedLearningWorkflow('multiple_config_not_exist.json', config_directory=self.cd)
        self.assertIsInstance(slw, mls.workflows.SupervisedLearningWorkflow)
        self.assertFalse(slw.task_terminated_init)

    def test_init_config_file_not_a_json_file_should_stop(self):
        """
        :test : mlsurvey.workflows.SupervisedLearningWorkflow()
        :condition : config file is not a json file
        :main_result : init is not terminated
        """
        slw = mls.workflows.SupervisedLearningWorkflow('config_loaded_not_json.json', config_directory=self.cd)
        self.assertIsInstance(slw, mls.workflows.SupervisedLearningWorkflow)
        self.assertFalse(slw.task_terminated_init)

    def test_init_SL_workflow_should_initialized_with_config(self):
        c = {'testconfig': 'config loaded'}
        slw = mls.workflows.SupervisedLearningWorkflow(config=c)
        self.assertEqual('config loaded', slw.config.data['testconfig'])

    def test_set_terminated_all_terminated(self):
        slw = mls.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertTrue(slw.terminated)

    def test_set_terminated_all_terminated_but_init(self):
        mlw = mls.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        mlw.task_terminated_init = False
        mlw.task_terminated_get_data = True
        mlw.task_terminated_prepare_data = True
        mlw.task_terminated_split_data = True
        mlw.task_terminated_learn = True
        mlw.task_terminated_evaluate = True
        mlw.task_terminated_persistence = True
        mlw.set_terminated()
        self.assertFalse(mlw.terminated)

    def test_set_terminated_all_terminated_but_get_data(self):
        slw = mls.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = False
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_prepare_data(self):
        slw = mls.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = False
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_split_data(self):
        slw = mls.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = False
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_learn(self):
        slw = mls.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = False
        slw.task_terminated_evaluate = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_evaluate(self):
        slw = mls.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = False
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_persistence(self):
        slw = mls.workflows.SupervisedLearningWorkflow(config_directory=self.cd)
        slw.task_terminated_init = True
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_persistence = False
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_task_get_data_data_is_obtained(self):
        slw = mls.workflows.SupervisedLearningWorkflow('complete_config_loaded.json', config_directory=self.cd)
        slw.task_get_data()
        self.assertIsInstance(slw.context.dataset, mls.datasets.NClassRandomClassificationWithNoise)
        self.assertIsNotNone(slw.context.data)
        self.assertEqual(100, len(slw.context.data.x))
        self.assertEqual(100, len(slw.context.data.y))
        self.assertTrue(slw.task_terminated_get_data)

    def test_task_prepare_data_data_should_be_prepared(self):
        slw = mls.workflows.SupervisedLearningWorkflow('complete_config_loaded.json', config_directory=self.cd)
        slw.task_get_data()
        self.assertEqual(-1.766054694735782, slw.context.data.x[0][0])
        slw.task_prepare_data()
        self.assertEqual(-0.7655005998158294, slw.context.data.x[0][0])
        self.assertTrue(slw.task_terminated_prepare_data)

    def test_task_split_data_data_should_be_split_train_test(self):
        slw = mls.workflows.SupervisedLearningWorkflow('complete_config_loaded.json', config_directory=self.cd)
        slw.task_get_data()
        self.assertEqual(100, len(slw.context.data.x))
        slw.task_split_data()
        self.assertEqual(100, len(slw.context.data.x))
        self.assertEqual(20, len(slw.context.data_test.x))
        self.assertEqual(20, len(slw.context.data_test.y))
        self.assertEqual(80, len(slw.context.data_train.x))
        self.assertEqual(80, len(slw.context.data_train.y))
        self.assertTrue(slw.task_terminated_split_data)

    def test_task_learn_classifier_should_have_learn(self):
        slw = mls.workflows.SupervisedLearningWorkflow('complete_config_loaded.json', config_directory=self.cd)
        slw.task_get_data()
        slw.task_split_data()
        slw.task_learn()
        self.assertIsInstance(slw.context.classifier, neighbors.KNeighborsClassifier)
        # ADD some assert to test the learning
        self.assertTrue(slw.task_terminated_learn)

    def test_task_evaluate_classifier_should_have_evaluate(self):
        slw = mls.workflows.SupervisedLearningWorkflow('complete_config_loaded.json', config_directory=self.cd)
        slw.task_get_data()
        slw.task_split_data()
        slw.task_learn()
        slw.task_evaluate()
        self.assertEqual(0.95, slw.context.evaluation.score)
        self.assertTrue(slw.task_terminated_evaluate)

    def test_task_persist_data_classifier_should_have_been_saved(self):
        slw = mls.workflows.SupervisedLearningWorkflow('complete_config_loaded.json', config_directory=self.cd)
        slw.task_get_data()
        slw.task_split_data()
        slw.task_learn()
        slw.task_evaluate()
        slw.task_persist()
        self.assertTrue(os.path.isfile(slw.log.directory + 'config.json'))
        self.assertEqual('ae28ce16852d7a5ddbecdb07ea755339', mls.Utils.md5_file(slw.log.directory + 'config.json'))
        self.assertTrue(os.path.isfile(slw.log.directory + 'dataset.json'))
        self.assertEqual('66eafcadd6773bcf132096486a57263a', mls.Utils.md5_file(slw.log.directory + 'dataset.json'))
        self.assertTrue(os.path.isfile(slw.log.directory + 'input.json'))
        self.assertEqual('e6838d8be15c6567169bd3a9e4a3b290', mls.Utils.md5_file(slw.log.directory + 'input.json'))
        self.assertTrue(os.path.isfile(slw.log.directory + 'algorithm.json'))
        self.assertEqual('1697475bd77100f5a9c8806c462cbd0b', mls.Utils.md5_file(slw.log.directory + 'algorithm.json'))
        self.assertTrue(os.path.isfile(slw.log.directory + 'model.joblib'))
        self.assertEqual('ab69d79bc10e6456b689455ff420d21f', mls.Utils.md5_file(slw.log.directory + 'model.joblib'))
        self.assertTrue(os.path.isfile(slw.log.directory + 'evaluation.json'))
        self.assertEqual('086cb297d5a8d9207ae369343727568a', mls.Utils.md5_file(slw.log.directory + 'evaluation.json'))
        self.assertTrue(slw.task_terminated_persistence)

    def test_run_all_step_should_be_executed(self):
        slw = mls.workflows.SupervisedLearningWorkflow('complete_config_loaded.json', config_directory=self.cd)
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
        # persistence in done
        self.assertTrue(slw.task_terminated_persistence)

        # all tasks are finished
        self.assertTrue(slw.terminated)

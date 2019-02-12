import os
import shutil
import unittest

from sklearn import neighbors
from sklearn.preprocessing import StandardScaler

import mlsurvey as mls


class TestSupervisedLearningWorkflow(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir)

    def test_init_SL_workflow_should_initialized(self):
        slw = mls.SupervisedLearningWorkflow()
        self.assertIsNotNone(slw.config.data)
        self.assertFalse(slw.terminated)
        self.assertFalse(slw.task_terminated_get_data)
        self.assertFalse(slw.task_terminated_prepare_data)
        self.assertFalse(slw.task_terminated_split_data)
        self.assertFalse(slw.task_terminated_learn)
        self.assertFalse(slw.task_terminated_evaluate)
        self.assertFalse(slw.task_terminated_persistence)
        self.assertIsInstance(slw.data, mls.datasets.DataSet)
        self.assertIsInstance(slw.data_preparation, StandardScaler)
        self.assertIsInstance(slw.data_train, mls.Input)
        self.assertIsInstance(slw.data_test, mls.Input)
        self.assertIsNone(slw.classifier)
        self.assertEqual(0.0, slw.score)
        self.assertIsInstance(slw.log, mls.Logging)

    def test_set_terminated_all_terminated(self):
        slw = mls.SupervisedLearningWorkflow()
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertTrue(slw.terminated)

    def test_set_terminated_all_terminated_but_get_data(self):
        slw = mls.SupervisedLearningWorkflow()
        slw.task_terminated_get_data = False
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_prepare_data(self):
        slw = mls.SupervisedLearningWorkflow()
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = False
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_split_data(self):
        slw = mls.SupervisedLearningWorkflow()
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = False
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_learn(self):
        slw = mls.SupervisedLearningWorkflow()
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = False
        slw.task_terminated_evaluate = True
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_evaluate(self):
        slw = mls.SupervisedLearningWorkflow()
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = False
        slw.task_terminated_persistence = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_persistence(self):
        slw = mls.SupervisedLearningWorkflow()
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.task_terminated_persistence = False
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_task_get_data_data_is_obtained(self):
        slw = mls.SupervisedLearningWorkflow('complete_config_loaded.json')
        slw.task_get_data()
        self.assertIsNotNone(slw.data)
        self.assertEqual(100, len(slw.data.x))
        self.assertEqual(100, len(slw.data.y))
        self.assertTrue(slw.task_terminated_get_data)

    def test_task_prepare_data_data_should_be_prepared(self):
        slw = mls.SupervisedLearningWorkflow('complete_config_loaded.json')
        slw.task_get_data()
        self.assertEqual(-1.766054694735782, slw.data.x[0][0])
        slw.task_prepare_data()
        self.assertEqual(-0.7655005998158294, slw.data.x[0][0])
        self.assertTrue(slw.task_terminated_prepare_data)

    def test_task_split_data_data_should_be_split_train_test(self):
        slw = mls.SupervisedLearningWorkflow('complete_config_loaded.json')
        slw.task_get_data()
        self.assertEqual(100, len(slw.data.x))
        slw.task_split_data()
        self.assertEqual(20, len(slw.data_test.x))
        self.assertEqual(20, len(slw.data_test.y))
        self.assertEqual(80, len(slw.data_train.x))
        self.assertEqual(80, len(slw.data_train.y))
        self.assertTrue(slw.task_terminated_split_data)

    def test_task_learn_classifier_should_have_learn(self):
        slw = mls.SupervisedLearningWorkflow('complete_config_loaded.json')
        slw.task_get_data()
        slw.task_split_data()
        slw.task_learn()
        self.assertIsInstance(slw.classifier, neighbors.KNeighborsClassifier)
        # ADD some assert to test the learning
        self.assertTrue(slw.task_terminated_learn)

    def test_task_evaluate_classifier_should_have_evaluate(self):
        slw = mls.SupervisedLearningWorkflow('complete_config_loaded.json')
        slw.task_get_data()
        slw.task_split_data()
        slw.task_learn()
        slw.task_evaluate()
        self.assertEqual(0.95, slw.score)
        self.assertTrue(slw.task_terminated_evaluate)

    def test_task_persist_data_classifier_should_have_been_saved(self):
        slw = mls.SupervisedLearningWorkflow('complete_config_loaded.json')
        slw.task_get_data()
        slw.task_split_data()
        slw.task_learn()
        slw.task_evaluate()
        slw.task_persist()
        self.assertTrue(os.path.isfile(slw.log.directory + 'config.json'))
        self.assertEqual('8afaddc85b486762a84b7ab6969246c2', mls.Utils.md5_file(slw.log.directory + 'config.json'))
        self.assertTrue(os.path.isfile(slw.log.directory + 'input.json'))
        self.assertEqual('0b62a3861adcbdb1a2c68541e8a1519b', mls.Utils.md5_file(slw.log.directory + 'input.json'))
        self.assertTrue(os.path.isfile(slw.log.directory + 'model.joblib'))
        self.assertEqual('ab69d79bc10e6456b689455ff420d21f', mls.Utils.md5_file(slw.log.directory + 'model.joblib'))
        self.assertTrue(os.path.isfile(slw.log.directory + 'evaluation.json'))
        self.assertEqual('6fd40c49cbdfd2fba040ace1a51ef989', mls.Utils.md5_file(slw.log.directory + 'evaluation.json'))
        self.assertTrue(slw.task_terminated_persistence)

    def test_load_data_classifier_loaded(self):
        slw = mls.SupervisedLearningWorkflow()
        slw.load_data_classifier("files/slw/")
        self.assertTrue('DataSet1', slw.config.data.learning_process.input)
        self.assertEqual(20, len(slw.data_test.x))
        self.assertEqual(20, len(slw.data_test.y))
        self.assertEqual(80, len(slw.data_train.x))
        self.assertEqual(80, len(slw.data_train.y))
        self.assertIsInstance(slw.classifier, neighbors.KNeighborsClassifier)
        self.assertEqual(0.95, slw.score)

    def test_run_all_step_should_be_executed(self):
        slw = mls.SupervisedLearningWorkflow('complete_config_loaded.json')
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

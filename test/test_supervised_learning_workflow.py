import unittest

from sklearn import neighbors
from sklearn.preprocessing import StandardScaler

import mlsurvey as mls


class TestSupervisedLearningWorkflow(unittest.TestCase):

    def test_init_SL_workflow_should_initialized(self):
        slw = mls.SupervisedLearningWorkflow()
        self.assertIsNotNone(slw.config.data)
        self.assertFalse(slw.terminated)
        self.assertFalse(slw.task_terminated_get_data)
        self.assertFalse(slw.task_terminated_prepare_data)
        self.assertFalse(slw.task_terminated_split_data)
        self.assertFalse(slw.task_terminated_learn)
        self.assertFalse(slw.task_terminated_evaluate)
        self.assertIsInstance(slw.data, mls.datasets.DataSet)
        self.assertIsInstance(slw.data_preparation, StandardScaler)
        self.assertIsInstance(slw.data_train, mls.Input)
        self.assertIsInstance(slw.data_test, mls.Input)
        self.assertIsNone(slw.classifier)
        self.assertEqual(0.0, slw.score)

    def test_set_terminated_all_terminated(self):
        slw = mls.SupervisedLearningWorkflow()
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.set_terminated()
        self.assertTrue(slw.terminated)

    def test_set_terminated_all_terminated_but_get_data(self):
        slw = mls.SupervisedLearningWorkflow()
        slw.task_terminated_get_data = False
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_prepare_data(self):
        slw = mls.SupervisedLearningWorkflow()
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = False
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_split_data(self):
        slw = mls.SupervisedLearningWorkflow()
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = False
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_learn(self):
        slw = mls.SupervisedLearningWorkflow()
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = False
        slw.task_terminated_evaluate = True
        slw.set_terminated()
        self.assertFalse(slw.terminated)

    def test_set_terminated_all_terminated_but_evaluate(self):
        slw = mls.SupervisedLearningWorkflow()
        slw.task_terminated_get_data = True
        slw.task_terminated_prepare_data = True
        slw.task_terminated_split_data = True
        slw.task_terminated_learn = True
        slw.task_terminated_evaluate = False
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

    def test_task_learn_classifier_should_have_evaluate(self):
        slw = mls.SupervisedLearningWorkflow('complete_config_loaded.json')
        slw.task_get_data()
        slw.task_split_data()
        slw.task_learn()
        slw.task_evaluate()
        self.assertEqual(0.95, slw.score)
        self.assertTrue(slw.task_terminated_evaluate)

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

        # all tasks are finished
        self.assertTrue(slw.terminated)

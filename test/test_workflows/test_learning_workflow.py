import unittest
import os

import mlsurvey as mls


class TestLearningWorkflow(unittest.TestCase):

    def test_visualize_class(self):
        self.assertEqual(mls.workflows.LearningWorkflow.visualize_class(), mls.visualize.VisualizeLogDetail)

    def test_init_should_init(self):
        lw = mls.workflows.LearningWorkflow()
        self.assertFalse(lw.terminated)
        self.assertTrue(lw.task_terminated_init)
        self.assertEqual('config/', lw.config_directory)
        self.assertEqual('', lw.base_directory)
        self.assertEqual(lw.config_file, 'config.json')
        self.assertIsInstance(lw.log, mls.Logging)
        self.assertEqual(lw.log.base_dir, 'logs/')
        self.assertIsNotNone(lw.log.mlflow_run_id)

    def test_init_with_confdir_should_init(self):
        lw = mls.workflows.LearningWorkflow(config_directory='test/')
        self.assertFalse(lw.terminated)
        self.assertTrue(lw.task_terminated_init)
        self.assertEqual('test/', lw.config_directory)

    def test_init_with_basedir_should_init(self):
        lw = mls.workflows.LearningWorkflow(base_directory='/')
        self.assertFalse(lw.terminated)
        self.assertTrue(lw.task_terminated_init)
        self.assertEqual('config/', lw.config_directory)
        self.assertEqual('/', lw.base_directory)

    def test_init_with_conffile_should_init(self):
        lw = mls.workflows.LearningWorkflow(config_file='conf.json')
        self.assertFalse(lw.terminated)
        self.assertTrue(lw.task_terminated_init)
        self.assertEqual('conf.json', lw.config_file)

    def test_init_with_logdir_should_init(self):
        """
        :test : mlsurvey.workflows.LearningWorkflow()
        :condition : set the logging directory
        :main_result : logging directory is set as specified
        """
        expected_dir = 'testlog/'
        slw = mls.workflows.LearningWorkflow(logging_dir=expected_dir)
        self.assertEqual(os.path.join('logs/', expected_dir), slw.log.directory)

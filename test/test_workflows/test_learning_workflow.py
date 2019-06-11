import unittest

import mlsurvey as mls


class TestLearningWorkflow(unittest.TestCase):

    def test_init_should_init(self):
        lw = mls.workflows.LearningWorkflow()
        self.assertFalse(lw.terminated)
        self.assertTrue(lw.task_terminated_init)
        self.assertEqual('config/', lw.config_directory)
        self.assertEqual('', lw.base_directory)

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

import unittest

import mlsurvey as mls


class TestMultipleLearningWorkflow(unittest.TestCase):

    def test_init_multiple_learning_workflow_should_initialized(self):
        mlw = mls.MultipleLearningWorkflow('multiple_config.json')
        self.assertIsNotNone(mlw.config.data)
        self.assertTrue(mlw.task_terminated_expand_config)

    def test_set_terminated_all_terminated(self):
        mlw = mls.MultipleLearningWorkflow()
        mlw.task_terminated_expand_config = True
        mlw.set_terminated()
        self.assertTrue(mlw.terminated)

    def test_set_terminated_all_terminated_but_expand_config(self):
        mlw = mls.MultipleLearningWorkflow()
        mlw.task_terminated_expand_config = False
        mlw.set_terminated()
        self.assertFalse(mlw.terminated)

    def test_run_all_step_should_be_executed(self):
        mlw = mls.MultipleLearningWorkflow('multiple_config.json')
        self.assertFalse(mlw.terminated)
        mlw.run()

        self.assertEqual(3, len(mlw.expanded_config))
        # TODO to continue

        # all tasks are finished
        self.assertTrue(mlw.terminated)

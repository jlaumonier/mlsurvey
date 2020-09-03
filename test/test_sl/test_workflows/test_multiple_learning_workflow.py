import os
import shutil
import unittest

import mlsurvey as mls


class TestMultipleLearningWorkflow(unittest.TestCase):
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

    def test_init_multiple_learning_workflow_should_initialized(self):
        mlw = mls.sl.workflows.MultipleLearningWorkflow('multiple_config.json', config_directory=self.cd)
        self.assertIsNotNone(mlw.config.data)
        self.assertEqual(self.cd, mlw.config_directory)
        self.assertFalse(mlw.task_terminated_expand_config)
        self.assertFalse(mlw.task_terminated_run_each_config)

    def test_init_config_file_not_exists_should_be_not_terminated(self):
        """
        :test : mlsurvey.workflows.MultipleLearningWorkflow()
        :condition : config file not exists
        :main_result : init is not terminated
        """
        mlw = mls.sl.workflows.MultipleLearningWorkflow('multiple_config_not_exist.json', config_directory=self.cd)
        self.assertIsInstance(mlw, mls.sl.workflows.MultipleLearningWorkflow)
        self.assertFalse(mlw.task_terminated_init)

    def test_init_config_file_not_a_json_file_should_stop(self):
        """
        :test : mlsurvey.workflows.MultipleLearningWorkflow()
        :condition : config file is not a json file
        :main_result : init is not terminated
        """
        mlw = mls.sl.workflows.MultipleLearningWorkflow('config_loaded_not_json.json', config_directory=self.cd)
        self.assertIsInstance(mlw, mls.sl.workflows.MultipleLearningWorkflow)
        self.assertFalse(mlw.task_terminated_init)

    def test_run_each_config_all_should_be_ran(self):
        mlw = mls.sl.workflows.MultipleLearningWorkflow(config_file='multiple_config.json', config_directory=self.cd)
        mlw.task_expand_config()
        mlw.task_run_each_config()
        self.assertEqual(3, len(mlw.slw))
        self.assertTrue(mlw.slw[0].terminated)
        self.assertTrue(mlw.slw[1].terminated)
        self.assertTrue(mlw.slw[2].terminated)
        self.assertTrue(mlw.task_terminated_run_each_config)

    def test_run_all_step_should_be_executed(self):
        mlw = mls.sl.workflows.MultipleLearningWorkflow('multiple_config.json', config_directory=self.cd)
        self.assertFalse(mlw.terminated)
        mlw.run()

        # expand task
        self.assertTrue(mlw.task_terminated_expand_config)
        # run each config task
        self.assertTrue(mlw.task_terminated_run_each_config)

        # all tasks are finished
        self.assertTrue(mlw.terminated)

    def test_run_all_step_init_not_terminated_should_not_be_executed(self):
        """
        :test : mlsurvey.workflows.MultipleLearningWorkflow.run()
        :condition : config file not exists
        :main_result : should not run
        """
        mlw = mls.sl.workflows.MultipleLearningWorkflow('multiple_config_not_exist.json', config_directory=self.cd)
        self.assertFalse(mlw.terminated)
        mlw.run()
        self.assertFalse(mlw.terminated)

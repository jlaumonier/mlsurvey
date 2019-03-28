import os
import shutil
import unittest

import mlsurvey as mls


class TestMultipleLearningWorkflow(unittest.TestCase):
    cd = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.cd = os.path.join(directory, '../config/')

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir)

    def test_init_multiple_learning_workflow_should_initialized(self):
        mlw = mls.workflows.MultipleLearningWorkflow('multiple_config.json', config_directory=self.cd)
        self.assertIsNotNone(mlw.config.data)
        self.assertFalse(mlw.task_terminated_expand_config)
        self.assertFalse(mlw.task_terminated_run_each_config)

    def test_set_terminated_all_terminated(self):
        mlw = mls.workflows.MultipleLearningWorkflow(config_directory=self.cd)
        mlw.task_terminated_expand_config = True
        mlw.task_terminated_run_each_config = True
        mlw.set_terminated()
        self.assertTrue(mlw.terminated)

    def test_set_terminated_all_terminated_but_expand_config(self):
        mlw = mls.workflows.MultipleLearningWorkflow(config_directory=self.cd)
        mlw.task_terminated_expand_config = False
        mlw.task_terminated_run_each_config = True
        mlw.set_terminated()
        self.assertFalse(mlw.terminated)

    def test_set_terminated_all_terminated_but_run_each_config(self):
        mlw = mls.workflows.MultipleLearningWorkflow(config_directory=self.cd)
        mlw.task_terminated_expand_config = True
        mlw.task_terminated_run_each_config = False
        mlw.set_terminated()
        self.assertFalse(mlw.terminated)

    def test_task_expand_config_input_should_have_expanded(self):
        mlw = mls.workflows.MultipleLearningWorkflow('multiple_config.json', config_directory=self.cd)
        mlw.task_expand_config()
        self.assertEqual(3, len(mlw.expanded_config))
        d0 = {"input": "DataSet1", "split": "traintest20", "algorithm": "knn-base"}
        d1 = {"input": "DataSet2", "split": "traintest20", "algorithm": "knn-base"}
        d2 = {"input": "DataSet3", "split": "traintest20", "algorithm": "knn-base"}
        self.assertDictEqual(d0, mlw.expanded_config[0]['learning_process'])
        self.assertDictEqual(d1, mlw.expanded_config[1]['learning_process'])
        self.assertDictEqual(d2, mlw.expanded_config[2]['learning_process'])
        self.assertEqual('NClassRandomClassificationWithNoise', mlw.expanded_config[0]['datasets']['DataSet1']['type'])
        self.assertEqual(100, mlw.expanded_config[0]['datasets']['DataSet1']['parameters']['n_samples'])
        self.assertEqual('make_circles', mlw.expanded_config[1]['datasets']['DataSet2']['type'])
        self.assertEqual(100, mlw.expanded_config[1]['datasets']['DataSet2']['parameters']['n_samples'])
        self.assertEqual('load_iris', mlw.expanded_config[2]['datasets']['DataSet3']['type'])
        self.assertEqual(0, len(mlw.expanded_config[2]['datasets']['DataSet3']['parameters']))
        self.assertTrue(mlw.task_terminated_expand_config)

    def test_task_expand_config_all_should_have_expanded(self):
        mlw = mls.workflows.MultipleLearningWorkflow('full_multiple_config.json', config_directory=self.cd)
        mlw.task_expand_config()
        self.assertEqual(36, len(mlw.expanded_config))
        lp0 = {"input": "DataSet1", "split": "traintest20", "algorithm": "knn-base"}
        ds0 = {"type": "make_classification",
               "parameters": {
                   "n_samples": 100,
                   "shuffle": True,
                   "noise": 0,
                   "random_state": 0
               }
               }
        self.assertDictEqual(lp0, mlw.expanded_config[0]['learning_process'])
        self.assertDictEqual(ds0, mlw.expanded_config[0]['datasets']['DataSet1'])
        self.assertEqual(1, len(mlw.expanded_config[0]['datasets']))

    def test_run_each_config_all_should_be_ran(self):
        mlw = mls.workflows.MultipleLearningWorkflow('multiple_config.json', config_directory=self.cd)
        mlw.task_expand_config()
        mlw.task_run_each_config()
        self.assertEqual(3, len(mlw.slw))
        self.assertTrue(mlw.slw[0].terminated)
        self.assertTrue(mlw.slw[1].terminated)
        self.assertTrue(mlw.slw[2].terminated)
        self.assertTrue(mlw.task_terminated_run_each_config)

    def test_run_all_step_should_be_executed(self):
        mlw = mls.workflows.MultipleLearningWorkflow('multiple_config.json', config_directory=self.cd)
        self.assertFalse(mlw.terminated)
        mlw.run()

        # expand task
        self.assertTrue(mlw.task_terminated_expand_config)
        # run each config task
        self.assertTrue(mlw.task_terminated_run_each_config)

        # all tasks are finished
        self.assertTrue(mlw.terminated)

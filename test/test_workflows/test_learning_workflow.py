import os
import shutil
import unittest

import mlsurvey as mls


class TestLearningWorkflow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.base_directory = os.path.join(directory, '../')

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir, ignore_errors=True)

    def test_visualize_class(self):
        self.assertEqual(mls.workflows.LearningWorkflow.visualize_class(), mls.visualize.VisualizeLogDetail)

    def test_init_should_init(self):
        lw = mls.workflows.LearningWorkflow(base_directory=self.base_directory)
        self.assertFalse(lw.terminated)
        self.assertTrue(lw.task_terminated_init)
        self.assertEqual('config/', lw.config_directory)
        self.assertEqual(self.base_directory, lw.base_directory)
        self.assertEqual(lw.config_file, 'config.json')
        self.assertIsInstance(lw.log, mls.Logging)
        self.assertEqual(lw.log.base_dir, os.path.join(self.base_directory, 'logs'))
        self.assertIsNone(lw.sources_directories)

    def test_init_with_confdir_should_init(self):
        lw = mls.workflows.LearningWorkflow(base_directory=self.base_directory,
                                            config_directory='files/')
        self.assertFalse(lw.terminated)
        self.assertTrue(lw.task_terminated_init)
        self.assertEqual('files/', lw.config_directory)

    def test_init_with_conffile_should_init(self):
        lw = mls.workflows.LearningWorkflow(base_directory=self.base_directory,
                                            config_file='complete_config_loaded.json')
        self.assertFalse(lw.terminated)
        self.assertTrue(lw.task_terminated_init)
        self.assertEqual('complete_config_loaded.json', lw.config_file)

    def test_init_with_confdict_should_init(self):
        config_dict = {"testconfig": "config loaded"}
        lw = mls.workflows.LearningWorkflow(base_directory=self.base_directory,
                                            config_dict=config_dict)
        self.assertEqual(config_dict, lw.config.data)

    def test_init_with_logdir_should_init(self):
        """
        :test : mlsurvey.workflows.LearningWorkflow()
        :condition : set the logging directory
        :main_result : logging directory is set as specified
        """
        expected_dir = 'testlog/'
        slw = mls.workflows.LearningWorkflow(base_directory=self.base_directory,
                                             logging_dir=expected_dir)
        self.assertEqual(os.path.join(self.base_directory, 'logs/', expected_dir), slw.log.directory)

    def test_init_with_sources_directories_should_init(self):
        """
        :test : mlsurvey.workflows.LearningWorkflow()
        :condition : set sources directories
        :main_result : sources directories are initialized and copied inside the log directory
        """
        slw = mls.workflows.LearningWorkflow(base_directory=self.base_directory,
                                             sources_directories={'source1': os.path.join(self.base_directory,
                                                                                          'files/slw'),
                                                                  'source2': os.path.join(self.base_directory,
                                                                                          'files/config.json')})
        self.assertEqual(2, len(slw.sources_directories))
        self.assertEqual(2, len([i for i in os.listdir(os.path.join(slw.log.directory, 'sources'))]))
        self.assertTrue(os.path.isdir(os.path.join(slw.log.directory, 'sources', 'source1', 'slw', 'EvaluateTask')))
        self.assertTrue(os.path.isfile(os.path.join(slw.log.directory, 'sources', 'source1', 'slw', 'config.json')))
        self.assertTrue(os.path.isfile(os.path.join(slw.log.directory, 'sources', 'source2', 'config.json')))

    def test_init_with_mlflow_mlflow_should_init(self):
        """
        :test : mlsurvey.workflows.LearningWorkflow()
        :condition : init mlflow without xp_name
        :main_result : mlflow is initialized and xp name is set correctly
        """
        slw = mls.workflows.LearningWorkflow(base_directory=self.base_directory,
                                             mlflow_log=True)
        self.assertEqual(slw.log.mlflow_experiment.name, slw.log.dir_name)
        self.assertIsNotNone(slw.log.mlflow_current_run)

    def test_terminate(self):
        """
        :test : mlsurvey.workflows.LearningWorkflow.terminate()
        :condition : -
        :main_result : mlflow is terminated and terminated.json is written
        """
        slw = mls.workflows.LearningWorkflow(base_directory=self.base_directory, mlflow_log=True)
        slw.terminate()
        self.assertTrue(slw.terminated)
        # should test but do not know how to test if mlflow run is terminated
        self.assertTrue(os.path.isfile(os.path.join(slw.log.base_dir, slw.log.dir_name, 'terminated.json')))

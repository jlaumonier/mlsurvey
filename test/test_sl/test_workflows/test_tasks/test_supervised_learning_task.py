import unittest
import os
import shutil

import mlsurvey as mls


class TestSupervisedLearningTask(unittest.TestCase):
    config_directory = ''
    base_directory = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.base_directory = os.path.join(directory, '../../../')
        cls.config_directory = os.path.join(cls.base_directory, 'config/')

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir, ignore_errors=True)

    def test_init_log_config(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LoadDataTask.init_log_config()
        :condition : -
        :main_result : Log and Config objects are initialized
        """
        log = mls.Logging()
        task = mls.sl.workflows.tasks.SupervisedLearningTask(logging_base_directory=log.base_dir,
                                                             logging_directory=log.dir_name,
                                                             config_filename='complete_config_loaded.json',
                                                             config_directory=self.config_directory)
        task.init_log_config()
        self.assertIsInstance(task.config, mls.Config)
        self.assertIsNotNone(task.config.data)
        self.assertIsInstance(task.log, mls.Logging)

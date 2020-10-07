import os
import shutil
import unittest

import mlsurvey as mls


class TestBaseTask(unittest.TestCase):
    config_directory = ''
    base_directory = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.base_directory = os.path.join(directory, '../../')
        cls.config_directory = 'config/'

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir, ignore_errors=True)

    def test_init_log_config_base_directory(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LoadDataTask.init_log_config()
        :condition : base directory is set
        :main_result : Log and Config objects are initialized
        """
        log = mls.Logging()
        task = mls.workflows.tasks.BaseTask(base_directory=self.base_directory,
                                            logging_base_directory=os.path.join(self.base_directory, log.base_dir),
                                            logging_directory=log.dir_name,
                                            config_filename='complete_config_loaded.json',
                                            config_directory=self.config_directory)
        # config is initialized
        self.assertIsInstance(task.config, mls.Config)
        # config is correctly loaded
        self.assertIsNotNone(task.config.data)
        # ensure that all refs are replaced into the config data
        self.assertTrue(task.config.is_compacted())
        # log is initialized
        self.assertIsInstance(task.log, mls.Logging)
        # log base dir is correctly initalized
        self.assertEqual(os.path.join(self.base_directory, 'logs/', ), task.log.base_dir)
        # log directory is correctly initalized
        self.assertEqual(os.path.join(self.base_directory, 'logs/', task.log.dir_name), task.log.directory)

    def test_init_log_config_with_already_compacted_config(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LoadDataTask.init_log_config()
        :condition : config file is already compatected (ex : from multiple wf)
        :main_result : Config objects are initialized (Log not tested here
        """
        log = mls.Logging()
        task = mls.workflows.tasks.BaseTask(base_directory=self.base_directory,
                                            logging_base_directory=os.path.join(self.base_directory, log.base_dir),
                                            logging_directory=log.dir_name,
                                            config_filename='compacted_config.json',
                                            config_directory=self.config_directory)
        # config is initialized
        self.assertIsInstance(task.config, mls.Config)
        # config is correctly loaded
        self.assertIsNotNone(task.config.data)
        # ensure that all refs are replaced into the config data
        self.assertTrue(task.config.is_compacted())

    def test_init_log_config_base_directory_not_loaded(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LoadDataTask.init_log_config()
        :condition : base directory is set with wrong directory
        :main_result : Log and Config objects are not initialized
        """
        log = mls.Logging()
        try:
            _ = mls.workflows.tasks.BaseTask(base_directory=self.base_directory,
                                             logging_base_directory=os.path.join(self.base_directory, log.base_dir),
                                             logging_directory=log.dir_name,
                                             config_filename='not_existed_file.json',
                                             config_directory=self.config_directory)
            self.assertTrue(False)
        except FileNotFoundError:
            self.assertTrue(True)

    def test_init_log_config_config_not_json(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LoadDataTask.init_log_config()
        :condition : config file is not a json file
        :main_result : Log and Config objects are not initialized
        """
        log = mls.Logging()
        try:
            _ = mls.workflows.tasks.BaseTask(base_directory=self.base_directory,
                                             logging_base_directory=os.path.join(self.base_directory, log.base_dir),
                                             logging_directory=log.dir_name,
                                             config_filename='config_loaded_not_json.json',
                                             config_directory=self.config_directory)
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

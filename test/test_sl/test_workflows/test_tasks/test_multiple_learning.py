import os
import shutil
import unittest

import luigi

import mlsurvey as mls


class TestMultipleLearningTask(unittest.TestCase):
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
        shutil.rmtree(os.path.join(cls.base_directory, log.base_dir), ignore_errors=True)

    def test_run(self):
        """
        :test : mlsurvey.sl.workflows.tasks.TestMultipleLearningTask.run()
        :condition : config file contains multiple learning config
        :main_result : all learning have ran
        """
        temp_log = mls.Logging()
        luigi.build([mls.sl.workflows.tasks.MultipleLearningTask(logging_directory=temp_log.dir_name,
                                                                 logging_base_directory=os.path.join(
                                                                     self.base_directory,
                                                                     temp_log.base_dir),
                                                                 config_filename='multiple_config.json',
                                                                 config_directory=self.config_directory,
                                                                 base_directory=self.base_directory)],
                    local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'results.json')))
        result_dict = log.load_json_as_dict('results.json')
        self.assertEqual(3, result_dict['NbLearning'])


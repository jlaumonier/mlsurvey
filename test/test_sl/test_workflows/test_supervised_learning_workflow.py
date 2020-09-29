import os
import shutil
import unittest

import mlsurvey as mls


class TestSupervisedLearningWorkflow(unittest.TestCase):
    cd = ''
    base_directory = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.cd = os.path.join(directory, '../../config/')
        cls.base_directory = os.path.join(directory, '../../')

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(os.path.join(cls.base_directory, log.base_dir), ignore_errors=True)

    def test_run_all_step_should_be_executed(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow('complete_config_loaded.json',
                                                          config_directory=self.cd,
                                                          base_directory=self.base_directory)
        slw.run()
        self.assertTrue(os.path.isfile(os.path.join(self.base_directory, slw.log.directory, 'config.json')))
        self.assertTrue(os.path.isfile(os.path.join(self.base_directory, slw.log.directory, 'dataset.json')))
        self.assertTrue(os.path.isfile(os.path.join(self.base_directory, slw.log.directory, 'raw_data.json')))
        self.assertTrue(os.path.isfile(os.path.join(self.base_directory, slw.log.directory, 'raw_data-content.h5')))
        self.assertTrue(os.path.isfile(os.path.join(self.base_directory, slw.log.directory, 'data.json')))
        self.assertTrue(os.path.isfile(os.path.join(self.base_directory, slw.log.directory, 'data-content.h5')))
        self.assertTrue(os.path.isfile(os.path.join(self.base_directory, slw.log.directory, 'data.json')))
        self.assertTrue(os.path.isfile(os.path.join(self.base_directory, slw.log.directory, 'data-content.h5')))
        self.assertTrue(os.path.isfile(os.path.join(self.base_directory, slw.log.directory, 'split_data.json')))
        self.assertTrue(os.path.isfile(os.path.join(self.base_directory, slw.log.directory, 'train-content.h5')))
        self.assertTrue(os.path.isfile(os.path.join(self.base_directory, slw.log.directory, 'test-content.h5')))
        self.assertTrue(os.path.isfile(os.path.join(self.base_directory, slw.log.directory, 'raw_train-content.h5')))
        self.assertTrue(os.path.isfile(os.path.join(self.base_directory, slw.log.directory, 'raw_test-content.h5')))
        self.assertTrue(os.path.isfile(os.path.join(self.base_directory, slw.log.directory, 'evaluation.json')))

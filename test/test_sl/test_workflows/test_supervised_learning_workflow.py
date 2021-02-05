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

    def test_visualize_class(self):
        self.assertEqual(mls.sl.workflows.SupervisedLearningWorkflow.visualize_class(), mls.sl.visualize.VisualizeLogSL)

    def test_run_all_step_should_be_executed(self):
        slw = mls.sl.workflows.SupervisedLearningWorkflow('complete_config_loaded.json',
                                                          config_directory=self.cd,
                                                          base_directory=self.base_directory)
        slw.run()
        bd = os.path.join(self.base_directory, slw.log.directory)
        self.assertTrue(os.path.isfile(os.path.join(bd, 'config.json')))
        self.assertTrue(os.path.isfile(os.path.join(bd, 'LoadDataTask', 'dataset.json')))
        self.assertTrue(os.path.isfile(os.path.join(bd, 'LoadDataTask', 'raw_data.json')))
        self.assertTrue(os.path.isfile(os.path.join(bd, 'LoadDataTask', 'raw_data-content.h5')))
        # self.assertTrue(os.path.isfile(os.path.join(bd, 'data.json')))
        # self.assertTrue(os.path.isfile(os.path.join(bd, 'data-content.h5')))
        # self.assertTrue(os.path.isfile(os.path.join(bd, 'data.json')))
        # self.assertTrue(os.path.isfile(os.path.join(bd, 'data-content.h5')))
        # self.assertTrue(os.path.isfile(os.path.join(bd, 'split_data.json')))
        # self.assertTrue(os.path.isfile(os.path.join(bd, 'train-content.h5')))
        # self.assertTrue(os.path.isfile(os.path.join(bd, 'test-content.h5')))
        # self.assertTrue(os.path.isfile(os.path.join(bd, 'raw_train-content.h5')))
        # self.assertTrue(os.path.isfile(os.path.join(bd, 'raw_test-content.h5')))
        # self.assertTrue(os.path.isfile(os.path.join(bd, 'evaluation.json')))

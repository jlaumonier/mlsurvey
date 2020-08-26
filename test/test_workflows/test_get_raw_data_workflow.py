import os
import shutil
import unittest

import luigi

import mlsurvey as mls


class TestGetRawDataWorkflow(unittest.TestCase):
    output = ''
    config_directory = ''
    base_directory = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.base_directory = os.path.join(directory, '../')
        cls.config_directory = 'config/'
        cls.output = 'output/'

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir, ignore_errors=True)
        shutil.rmtree(cls.output, ignore_errors=True)

    @unittest.skip('fill config_get_raw_data_wf.json with s3 parameter')
    def test_run(self):
        log = mls.Logging()
        luigi.build([mls.workflows.GetRawDataWorkflow(base_directory=self.base_directory,
                                                      logging_directory=log.dir_name,
                                                      logging_base_directory=log.base_dir,
                                                      config_filename='config_get_raw_data_wf.json',
                                                      config_directory=self.config_directory)], local_scheduler=True)
        expected_output = os.path.join(self.base_directory, self.output, 'test.txt')
        print(expected_output)
        self.assertTrue(os.path.isfile(expected_output))

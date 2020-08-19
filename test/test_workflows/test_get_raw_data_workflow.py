import unittest
import luigi
import os
import shutil

import mlsurvey as mls


class TestGetRawDataWorkflow(unittest.TestCase):
    output = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.output = os.path.join(directory, '../output/dataset')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output, ignore_errors=True)

    @unittest.skip("TODO : Implement an S3 test infrastructure")
    def test_run(self):
        luigi.build([mls.workflows.GetRawDataWorkflow(destination_path=self.output,
                                                      destination_name='test.txt',
                                                      source_path='test',
                                                      source_name='test.txt',
                                                      base_url='PUT_A_S3_URL_TO_TEST',
                                                      bucket_name='PUT_A_BUCKET_NAME_TO_TEST'
                                                      )], local_scheduler=True)
        expected_output = os.path.join(self.output, 'test.txt')
        self.assertTrue(os.path.isfile(expected_output))

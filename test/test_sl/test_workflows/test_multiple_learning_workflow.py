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

    def test_run_all_should_be_ran(self):
        """
        :test : mlsurvey.sl.workflows.MultipleLearningWorkflow.run()
        :condition : config file contains lists in fairness parameters
        :main_result : all learning have ran
        """
        mlw = mls.sl.workflows.MultipleLearningWorkflow(config_file='multiple_config.json', config_directory=self.cd)
        mlw.run()
        self.assertTrue(os.path.isfile(os.path.join(mlw.log.base_dir, mlw.log.dir_name, 'results.json')))
        result_dict = mlw.log.load_json_as_dict('results.json')
        self.assertEqual(3, result_dict['NbLearning'])

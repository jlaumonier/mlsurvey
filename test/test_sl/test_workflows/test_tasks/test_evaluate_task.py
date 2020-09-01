import os
import shutil
import unittest
import luigi
import numpy as np

import mlsurvey as mls


class TestEvaluateTask(unittest.TestCase):
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

    def test_run(self):
        """
        :test : mlsurvey.sl.workflows.tasks.EvaluateTask.run()
        :condition : model is trained
        :main_result : model is evaluated
        """
        log = mls.Logging()
        luigi.build([mls.sl.workflows.tasks.EvaluateTask(logging_directory=log.dir_name,
                                                         logging_base_directory=log.base_dir,
                                                         config_filename='complete_config_loaded.json',
                                                         config_directory=self.config_directory,
                                                         evaluation_type='mls.sl.models.EvaluationSupervised')],
                    local_scheduler=True)
        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'evaluation.json')))
        evaluation_dict = log.load_json_as_dict('evaluation.json')
        evaluation = mls.sl.models.EvaluationFactory.create_instance_from_dict(evaluation_dict)

        expected_cm = np.array([[13, 0], [1, 6]])
        self.assertEqual(0.95, evaluation.score)
        np.testing.assert_array_equal(expected_cm, evaluation.confusion_matrix)
        self.assertIsNone(evaluation.sub_evaluation)


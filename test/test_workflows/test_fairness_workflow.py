import os
import shutil
import unittest

import mlsurvey as mls


class TestFairnessWorkflow(unittest.TestCase):
    cd = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.cd = os.path.join(directory, '../config/')

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir)

    def test_init_should_init(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd)
        self.assertIsNotNone(fw.config.data)
        self.assertIsInstance(fw.context, mls.models.Context)
        self.assertIsInstance(fw.log, mls.Logging)
        self.assertFalse(fw.terminated)
        self.assertFalse(fw.task_terminated_get_data)
        self.assertFalse(fw.task_terminated_evaluate)
        self.assertFalse(fw.task_terminated_persistence)

    def test_set_terminated_all_terminated(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd)
        fw.task_terminated_get_data = True
        fw.task_terminated_evaluate = True
        fw.task_terminated_persistence = True
        fw.set_terminated()
        self.assertTrue(fw.terminated)

    def test_set_terminated_all_terminated_but_get_data(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd)
        fw.task_terminated_get_data = False
        fw.task_terminated_evaluate = True
        fw.task_terminated_persistence = True
        fw.set_terminated()
        self.assertFalse(fw.terminated)

    def test_set_terminated_all_terminated_but_evaluate(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd)
        fw.task_terminated_get_data = True
        fw.task_terminated_evaluate = False
        fw.task_terminated_persistence = True
        fw.set_terminated()
        self.assertFalse(fw.terminated)

    def test_set_terminated_all_terminated_but_persistence(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd)
        fw.task_terminated_get_data = True
        fw.task_terminated_evaluate = True
        fw.task_terminated_persistence = False
        fw.set_terminated()
        self.assertFalse(fw.terminated)

    def test_task_get_data_data_is_obtained(self):
        """
        :test : mls.workflows.FairnessWorkflow().task_get_data()
        :condition : config file contains fairness parameters
        :main_result : Generate and dataset contains the fairness parameters
        """
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd)
        fw.task_get_data()
        self.assertIsInstance(fw.context.dataset, mls.datasets.FileDataSet)
        self.assertEqual(12, fw.context.dataset.fairness['protected_attribute'])
        self.assertEqual("x >= 25", fw.context.dataset.fairness['privileged_classes'])
        self.assertIsNotNone(fw.context.data)
        self.assertEqual(1000, len(fw.context.data.x))
        self.assertEqual(1000, len(fw.context.data.y))
        self.assertTrue(fw.task_terminated_get_data)

    def test_task_get_data_no_fairness_data_error(self):
        """
        :test : mls.workflows.FairnessWorkflow().task_get_data()
        :condition : config file does not contains fairness parameters
        :main_result :
        """
        fw = mls.workflows.FairnessWorkflow(config_file='config_fairness_no_fairness_data.json',
                                            config_directory=self.cd)
        try:
            fw.task_get_data()
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_task_get_data_no_fairness_process(self):
        """
        :test : mls.workflows.FairnessWorkflow().task_get_data()
        :condition : config file does not contains fairness parameters
        :main_result :
        """
        fw = mls.workflows.FairnessWorkflow(config_file='complete_config_loaded.json', config_directory=self.cd)
        try:
            fw.task_get_data()
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_task_evaluate_classifier_should_have_evaluate(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd)
        fw.task_get_data()
        fw.task_evaluate()
        self.assertTrue(fw.task_terminated_evaluate)

    def test_task_persist_data_classifier_should_have_been_saved(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd)
        fw.task_get_data()
        fw.task_evaluate()
        fw.task_persist()
        self.assertTrue(os.path.isfile(fw.log.directory + 'config.json'))
        self.assertEqual('c39200b2e4983ee659e533c5d7b6fc21', mls.Utils.md5_file(fw.log.directory + 'config.json'))
        self.assertTrue(os.path.isfile(fw.log.directory + 'dataset.json'))
        self.assertEqual('11a1b140d634276101f08ce8f3da6ccc', mls.Utils.md5_file(fw.log.directory + 'dataset.json'))
        self.assertTrue(os.path.isfile(fw.log.directory + 'evaluation.json'))
        self.assertEqual('6bb9b7751380b79c1c8c1bb11c1ebc3d', mls.Utils.md5_file(fw.log.directory + 'evaluation.json'))
        self.assertTrue(fw.task_terminated_persistence)

    def test_run_all_step_should_be_executed(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd)
        self.assertFalse(fw.terminated)
        fw.run()

        # data is generated
        self.assertTrue(fw.task_terminated_get_data)
        # evaluation in done
        self.assertTrue(fw.task_terminated_evaluate)
        # persistence in done
        self.assertTrue(fw.task_terminated_persistence)

        # all tasks are finished
        self.assertTrue(fw.terminated)

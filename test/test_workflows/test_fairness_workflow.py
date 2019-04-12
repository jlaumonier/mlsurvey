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
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd)
        fw.task_get_data()
        self.assertIsInstance(fw.context.dataset, mls.datasets.GenericDataSet)
        self.assertEqual('make_circles', fw.context.dataset.t)
        self.assertIsNotNone(fw.context.data)
        self.assertEqual(100, len(fw.context.data.x))
        self.assertEqual(100, len(fw.context.data.y))
        self.assertTrue(fw.task_terminated_get_data)

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
        self.assertEqual('e88479b9097b80f044c68bcbfac8fe02', mls.Utils.md5_file(fw.log.directory + 'config.json'))
        self.assertTrue(os.path.isfile(fw.log.directory + 'dataset.json'))
        self.assertEqual('b032a00bb82348446accb17c964a2587', mls.Utils.md5_file(fw.log.directory + 'dataset.json'))
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

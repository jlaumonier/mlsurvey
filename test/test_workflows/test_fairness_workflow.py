import os
import shutil
import unittest

import mlsurvey as mls


class TestFairnessWorkflow(unittest.TestCase):
    cd = ''
    bd = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.cd = os.path.join(directory, '../config/')
        cls.bd = os.path.join(directory, '../')

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir, ignore_errors=True)

    def test_init_should_init(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd)
        self.assertIsNotNone(fw.config.data)
        self.assertFalse(fw.is_sub_process)
        self.assertIsNone(fw.parent_context)
        self.assertIsInstance(fw.context, mls.models.Context)
        self.assertIsInstance(fw.log, mls.Logging)
        self.assertFalse(fw.terminated)
        self.assertFalse(fw.task_terminated_get_data)
        self.assertFalse(fw.task_terminated_evaluate)
        self.assertFalse(fw.task_terminated_persistence)

    def test_init_should_init_with_existing_data(self):
        context = mls.models.Context(mls.models.EvaluationSupervised)
        fw = mls.workflows.FairnessWorkflow(context=context)
        self.assertIsNone(fw.config)
        self.assertTrue(fw.is_sub_process)
        self.assertEqual(context, fw.parent_context)

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

    def test_set_subprocess_terminated_all_terminated_but_persistence(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd)
        fw.task_terminated_get_data = True
        fw.task_terminated_evaluate = True
        fw.task_terminated_persistence = False
        fw.set_subprocess_terminated()
        self.assertTrue(fw.terminated)

    def test_set_subprocess_terminated_all_terminated_but_get_data(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd)
        fw.task_terminated_get_data = False
        fw.task_terminated_evaluate = True
        fw.task_terminated_persistence = True
        fw.set_subprocess_terminated()
        self.assertFalse(fw.terminated)

    def test_set_subprocess_terminated_all_terminated_but_evaluate(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd)
        fw.task_terminated_get_data = True
        fw.task_terminated_evaluate = False
        fw.task_terminated_persistence = True
        fw.set_subprocess_terminated()
        self.assertFalse(fw.terminated)

    def test_task_get_data_data_is_obtained_with_config(self):
        """
        :test : mls.workflows.FairnessWorkflow().task_get_data()
        :condition : config file contains fairness parameters
        :main_result : Generate and dataset contains the fairness parameters
        """
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd, base_directory=self.bd)
        fw.task_get_data()
        self.assertIsInstance(fw.context.dataset, mls.datasets.FileDataSet)
        self.assertEqual(1, fw.context.dataset.fairness['protected_attribute'])
        self.assertEqual("x >= 25", fw.context.dataset.fairness['privileged_classes'])
        self.assertIsNotNone(fw.context.data)
        self.assertEqual(13, len(fw.context.data.x))
        self.assertEqual(13, len(fw.context.data.y))
        self.assertTrue(fw.task_terminated_get_data)

    def test_task_get_data_data_is_obtained_with_context(self):
        """
        :test : mls.workflows.FairnessWorkflow().task_get_data()
        :condition : workflow is defined as subprocess and context contains fairness parameters
        :main_result : dataset and data are get from the context
        """
        context = mls.models.Context(mls.models.EvaluationSupervised)
        context.dataset = mls.datasets.DataSetFactory.create_dataset('FileDataSet')
        directory = os.path.join(self.bd, "files/dataset")
        dataset_params = {"directory": directory, "filename": "test-fairness.arff"}
        fairness_params = {"protected_attribute": 1, "privileged_classes": "x >= 25"}
        context.dataset.set_generation_parameters(dataset_params)
        context.dataset.set_fairness_parameters(fairness_params)
        context.raw_data = mls.models.Data(context.dataset.generate())
        fw = mls.workflows.FairnessWorkflow(context=context)
        fw.task_get_data()
        self.assertEqual(context.dataset, fw.context.dataset)
        self.assertEqual(context.dataset.fairness, fw.context.dataset.fairness)
        self.assertEqual(context.raw_data, fw.context.data)
        self.assertTrue(fw.task_terminated_get_data)

    def test_task_get_data_no_fairness_data_error_with_config(self):
        """
        :test : mls.workflows.FairnessWorkflow().task_get_data()
        :condition : config file does not contains fairness parameters
        :main_result : mls.exceptions.ConfigError is raised
        """
        fw = mls.workflows.FairnessWorkflow(config_file='config_fairness_no_fairness_data.json',
                                            config_directory=self.cd)
        try:
            fw.task_get_data()
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_task_get_data_no_fairness_data_error_with_context(self):
        """
        :test : mls.workflows.FairnessWorkflow().task_get_data()
        :condition : workflow is defined as subprocess and context does not contain fairness parameters
        :main_result : mls.exceptions.WorkflowError is raised
        """
        context = mls.models.Context(mls.models.EvaluationSupervised)
        context.dataset = mls.datasets.DataSetFactory.create_dataset('FileDataSet')
        directory = os.path.join(self.bd, "files/dataset")
        dataset_params = {"directory": directory, "filename": "test-fairness.arff"}
        context.dataset.set_generation_parameters(dataset_params)
        context.data = mls.models.Data(context.dataset.generate())
        fw = mls.workflows.FairnessWorkflow(context=context)
        try:
            fw.task_get_data()
            self.assertTrue(False)
        except mls.exceptions.WorkflowError:
            self.assertTrue(True)

    def test_task_get_data_no_fairness_process_with_config(self):
        """
        :test : mls.workflows.FairnessWorkflow().task_get_data()
        :condition : config file does not contains fairness parameters
        :main_result : mls.exceptions.ConfigError is raised
        """
        fw = mls.workflows.FairnessWorkflow(config_file='complete_config_loaded.json', config_directory=self.cd)
        try:
            fw.task_get_data()
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_task_evaluate_classifier_should_have_evaluate(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd, base_directory=self.bd)
        fw.task_get_data()
        fw.task_evaluate()
        self.assertAlmostEqual(-0.3666666, fw.context.evaluation.demographic_parity, delta=1e-07)
        self.assertIsNone(fw.context.evaluation.equal_opportunity)
        self.assertIsNone(fw.context.evaluation.statistical_parity)
        self.assertIsNone(fw.context.evaluation.average_equalized_odds)
        self.assertAlmostEqual(0.476190476, fw.context.evaluation.disparate_impact_rate, delta=1e-07)
        self.assertTrue(fw.task_terminated_evaluate)

    def test_run_with_real_data(self):
        fw = mls.workflows.FairnessWorkflow(config_file='config_fairness_all_data.json',
                                            config_directory=self.cd,
                                            base_directory=self.bd)
        fw.run()
        self.assertIsNotNone(fw.context.evaluation.demographic_parity)
        self.assertTrue(fw.terminated)

    def test_task_persist_data_classifier_should_have_been_saved(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd, base_directory=self.bd)
        fw.task_get_data()
        fw.task_evaluate()
        fw.task_persist()
        self.assertTrue(os.path.isfile(fw.log.directory + 'config.json'))
        self.assertEqual('03d9ab96b2677c8e7efcd8a063781472', mls.Utils.md5_file(fw.log.directory + 'config.json'))
        self.assertTrue(os.path.isfile(fw.log.directory + 'dataset.json'))
        self.assertEqual('974b0c85beea523957266b744c296d99', mls.Utils.md5_file(fw.log.directory + 'dataset.json'))
        self.assertTrue(os.path.isfile(fw.log.directory + 'evaluation.json'))
        self.assertEqual('8cbacb12d164526b675b9d819f6872d3', mls.Utils.md5_file(fw.log.directory + 'evaluation.json'))
        self.assertTrue(os.path.isfile(fw.log.directory + 'input.json'))
        self.assertEqual('f7c2c5e2c7a952d0b077bf8eec94a048', mls.Utils.md5_file(fw.log.directory + 'input.json'))
        self.assertTrue(fw.task_terminated_persistence)

    def test_run_all_step_should_be_executed(self):
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd, base_directory=self.bd)
        self.assertFalse(fw.terminated)
        fw.run()

        # data is generated
        self.assertTrue(fw.task_terminated_get_data)
        # evaluation is done
        self.assertTrue(fw.task_terminated_evaluate)
        # persistence is done
        self.assertTrue(fw.task_terminated_persistence)

        # all tasks are finished
        self.assertTrue(fw.terminated)

    def test_run_as_subprocess_all_step_but_persistence(self):
        """
        :test : mls.workflows.FairnessWorkflow().run_as_subprocess()
        :condition : config file exists
        :main_result : all tasks are executed except persistence
        """
        fw = mls.workflows.FairnessWorkflow(config_directory=self.cd, base_directory=self.bd)
        self.assertFalse(fw.terminated)
        fw.run_as_subprocess()

        # data is generated
        self.assertTrue(fw.task_terminated_get_data)
        # evaluation is done
        self.assertTrue(fw.task_terminated_evaluate)
        # persistence is not done
        self.assertFalse(fw.task_terminated_persistence)

        # all tasks are finished
        self.assertTrue(fw.terminated)

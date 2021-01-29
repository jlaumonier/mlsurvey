import os
import shutil
import unittest

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner
import mlflow

import mlsurvey as mls


class TestMultipleLearningTask(unittest.TestCase):
    config_directory = ''
    base_directory = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.base_directory = os.path.join(directory, '../../../')
        cls.config_directory = os.path.join(cls.base_directory, 'config/')
        cls.mlflow_client = mlflow.tracking.MlflowClient()
        cls.mlflow_experiments = cls.mlflow_client.list_experiments()

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(os.path.join(cls.base_directory, log.base_dir), ignore_errors=True)

    def _run_one_task(self, config_filename):
        # create node from Task
        expand_config_node = mls.sl.workflows.tasks.MultipleLearningTask.get_node()
        # Prepare a data catalog
        # TODO revoir gestion configuation et Logging
        final_config_directory = os.path.join(str(self.base_directory), str(self.config_directory))
        config = mls.Config(name=config_filename, directory=final_config_directory)
        config.compact()
        # init logging
        log = mls.Logging(base_dir=os.path.join(self.base_directory, 'logs'),
                          mlflow_log=True)

        data_catalog = DataCatalog({'config': MemoryDataSet(),
                                    'log': MemoryDataSet(),
                                    'expanded_config': MemoryDataSet()})
        data_catalog.save('config', config)
        data_catalog.save('log', log)
        # Assemble nodes into a pipeline
        pipeline = Pipeline([expand_config_node])
        # Create a runner to run the pipeline
        runner = SequentialRunner()
        # Run the pipeline
        runner.run(pipeline, data_catalog)
        return log, data_catalog

    def test_run(self):
        """
        :test : mlsurvey.sl.workflows.tasks.TestMultipleLearningTask.run()
        :condition : config file contains multiple learning config
        :main_result : all learning have ran
        """
        log, data_catalog = self._run_one_task('multiple_config.json')
        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'results.json')))
        result_dict = log.load_json_as_dict('results.json')
        self.assertEqual(3, result_dict['NbLearning'])

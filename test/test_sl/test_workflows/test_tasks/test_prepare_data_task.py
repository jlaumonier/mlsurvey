import os
import shutil

from kedro.pipeline.node import Node
from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner

import mlflow

import mlsurvey as mls


class TestPrepareDataTask(mls.testing.TaskTestCase):
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

    def test_get_node(self):
        """
        :test : mlsurvey.workflows.tasks.PrepareDataTask.get_node()
        :condition : -
        :main_result : create a kedro with input and output parameter
        """
        prepare_data_node = mls.sl.workflows.tasks.PrepareDataTask.get_node()
        self.assertIsInstance(prepare_data_node, Node)

    def test_prepare_data(self):
        """
        :test : mlsurvey.workflows.tasks.PrepareDataTask.prepare_data()
        :condition : -
        :main_result : load the data
        """
        config, log = self._init_config_log('complete_config_loaded.json',
                                            self.base_directory,
                                            self.config_directory)
        df_raw_data = mls.FileOperation.read_hdf('raw_data-content.h5',
                                                 os.path.join(self.base_directory, 'files/tasks/load_data'),
                                                 'Pandas')
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data)
        lx = len(raw_data.x)
        ly = len(raw_data.y)
        [prepared_data] = mls.sl.workflows.tasks.PrepareDataTask.prepare_data(config, log, raw_data)
        self.assertEqual(-0.7655005998158294, prepared_data.x[0][0])
        self.assertEqual(lx, len(prepared_data.x))
        self.assertEqual(ly, len(prepared_data.y))

    def _run_one_task(self, config_filename):
        # create node from Task
        load_data_node = mls.workflows.tasks.LoadDataTask.get_node()
        prepare_data_node = mls.sl.workflows.tasks.PrepareDataTask.get_node()

        config, log = self._init_config_log(config_filename, self.base_directory, self.config_directory)
        # Prepare a data catalog
        data_catalog = DataCatalog({'config': MemoryDataSet(),
                                    'log': MemoryDataSet(),
                                    'base_directory': MemoryDataSet()})
        data_catalog.save('config', config)
        data_catalog.save('log', log)
        data_catalog.save('base_directory', self.base_directory)
        # Assemble nodes into a pipeline
        pipeline = Pipeline([load_data_node, prepare_data_node])
        # Create a runner to run the pipeline
        runner = SequentialRunner()
        # Run the pipeline
        runner.run(pipeline, data_catalog)
        return log, config, data_catalog

    def test_run(self):
        """
        :test : mlsurvey.sl.workflows.tasks.PrepareDataTask.run()
        :condition : data file are loaded, saved in hdf database and logged
        :main_result : data are prepared.
        """
        config_filename = 'complete_config_loaded.json'
        log, config, data_catalog = self._run_one_task(config_filename)
        log.set_sub_dir(str(mls.sl.workflows.tasks.PrepareDataTask.__name__))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'data-content.h5')))
        df_data = mls.FileOperation.read_hdf('data-content.h5', os.path.join(log.directory), 'Pandas')
        data = mls.sl.models.DataFactory.create_data('Pandas', df_data)
        self.assertEqual(-0.7655005998158294, data.x[0][0])

    def test_run_prepare_textual_data(self):
        """
        :test : mlsurvey.sl.workflows.tasks.PrepareDataTask.run()
        :condition : data is textual
        :main_result : data are prepared.
        """
        config_filename = 'config_dataset_text.json'
        log, config, data_catalog = self._run_one_task(config_filename)

        log.set_sub_dir(str(mls.workflows.tasks.LoadDataTask.__name__))
        df_raw_data = mls.FileOperation.read_hdf('raw_data-content.h5',
                                                 os.path.join(log.directory),
                                                 'Pandas')
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data)
        lx = len(raw_data.x)
        ly = len(raw_data.y)
        self.assertEqual('7', raw_data.y[0])
        log.set_sub_dir(str(mls.sl.workflows.tasks.PrepareDataTask.__name__))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'data-content.h5')))
        df_data = mls.FileOperation.read_hdf('data-content.h5', os.path.join(log.directory), 'Pandas')
        data = mls.sl.models.DataFactory.create_data('Pandas', df_data)
        self.assertEqual(0.23989072176612425, data.x[0][0])
        self.assertEqual(lx, len(data.x))
        self.assertEqual(ly, len(data.y))


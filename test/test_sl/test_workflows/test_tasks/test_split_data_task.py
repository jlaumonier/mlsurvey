import os
import shutil

import mlflow

import mlsurvey as mls

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner
from kedro.pipeline.node import Node


class TestSplitDataTask(mls.testing.TaskTestCase):
    config_directory = ''
    base_directory = ''
    mlflow_client = None
    mlflow_experiments = None

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
        prepare_data_node = mls.sl.workflows.tasks.SplitDataTask.get_node()
        self.assertIsInstance(prepare_data_node, Node)

    def test_split_data(self):
        """
        :test : mlsurvey.sl.workflows.tasks.SplitDataTask.split_data()
        :condition : Data are prepared
        :main_result : data are split (train test) (prepared and raw).
        """
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
        df_prepared_data = mls.FileOperation.read_hdf('data-content.h5',
                                                      os.path.join(self.base_directory, 'files/tasks/prepare_data'),
                                                      'Pandas')
        prepared_data = mls.sl.models.DataFactory.create_data('Pandas', df_prepared_data)
        [train_data,
         test_data,
         train_raw_data,
         test_raw_data] = mls.sl.workflows.tasks.SplitDataTask.split_data(config, log, raw_data, prepared_data)
        self.assertEqual(100, len(prepared_data.x))
        self.assertEqual(100, len(prepared_data.y))
        self.assertEqual(20, len(test_data.x))
        self.assertEqual(20, len(test_data.y))
        self.assertEqual(80, len(train_data.x))
        self.assertEqual(80, len(train_data.y))
        self.assertEqual(80, len(train_raw_data.x))
        self.assertEqual(80, len(train_raw_data.y))
        self.assertEqual(20, len(test_raw_data.x))
        self.assertEqual(20, len(test_raw_data.y))
        self.assertEqual(prepared_data.df.shape[1], train_data.df.shape[1])
        self.assertEqual(prepared_data.df.shape[1], test_data.df.shape[1])
        self.assertEqual(prepared_data.df.shape[1], train_raw_data.df.shape[1])
        self.assertEqual(prepared_data.df.shape[1], test_raw_data.df.shape[1])
        # dfs should havec been reindexed
        self.assertEqual(len(train_data.df) - 1, train_data.df.index[len(train_data.df) - 1])
        self.assertEqual(len(test_data.df) - 1, train_data.df.index[len(test_data.df) - 1])
        self.assertEqual(len(train_raw_data.df) - 1, train_raw_data.df.index[len(train_raw_data.df) - 1])
        self.assertEqual(len(test_raw_data.df) - 1, train_raw_data.df.index[len(test_raw_data.df) - 1])

    def _run_one_task(self, config_filename):
        # create node from Task
        load_data_node = mls.workflows.tasks.LoadDataTask.get_node()
        prepare_data_node = mls.sl.workflows.tasks.PrepareDataTask.get_node()
        split_data_node = mls.sl.workflows.tasks.SplitDataTask.get_node()
        config, log = self._init_config_log(config_filename, self.base_directory, self.config_directory)
        # Prepare a data catalog
        data_catalog = DataCatalog({'config': MemoryDataSet(),
                                    'log': MemoryDataSet(),
                                    'base_directory': MemoryDataSet()})
        data_catalog.save('config', config)
        data_catalog.save('log', log)
        data_catalog.save('base_directory', self.base_directory)
        # Assemble nodes into a pipeline
        pipeline = Pipeline([load_data_node, prepare_data_node, split_data_node])
        # Create a runner to run the pipeline
        runner = SequentialRunner()
        # Run the pipeline
        runner.run(pipeline, data_catalog)
        return log, config, data_catalog

    def test_run(self):
        """
        :test : mlsurvey.sl.workflows.tasks.SplitDataTask.run()
        :condition : Data are prepared, saved in hdf database and logged
        :main_result : data are split (train test).
        """
        config_filename = 'complete_config_loaded.json'
        log, config, data_catalog = self._run_one_task(config_filename)
        log.set_sub_dir(str(mls.sl.workflows.tasks.SplitDataTask.__name__))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'train-content.h5')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'test-content.h5')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'raw_train-content.h5')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'raw_test-content.h5')))

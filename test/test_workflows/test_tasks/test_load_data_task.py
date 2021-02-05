import os
import shutil

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.runner import SequentialRunner
import mlflow

import mlsurvey as mls


class TestLoadDataTask(mls.testing.TaskTestCase):
    config_directory = ''
    base_directory = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.base_directory = os.path.join(directory, '../../')
        cls.config_directory = os.path.join(cls.base_directory, 'config/')
        cls.mlflow_client = mlflow.tracking.MlflowClient()
        cls.mlflow_experiments = cls.mlflow_client.list_experiments()

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(os.path.join(cls.base_directory, log.base_dir), ignore_errors=True)

    def test_init_dataset(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LoadDataTask.init_dataset()
        :condition : params set as defined
        :main_result : dataset is initialized
        """
        ds_param_dict = {"n_samples": 100,
                         "shuffle": True,
                         "random_state": 0,
                         "noise": 0
                         }
        ds_dict = {'type': 'NClassRandomClassificationWithNoise',
                   'parameters': ds_param_dict
                   }

        dataset = mls.workflows.tasks.LoadDataTask.init_dataset(ds_dict)
        self.assertIsInstance(dataset, mls.sl.datasets.NClassRandomClassificationWithNoise)
        self.assertDictEqual(dataset.params, ds_param_dict)
        self.assertDictEqual({}, dataset.fairness)

    def test_load_data(self):
        """
        :test : mlsurvey.workflows.tasks.LoadDataTask.load_data()
        :condition : -
        :main_result : load the data
        """
        config, log = self._init_config_log('complete_config_loaded.json',
                                            self.base_directory,
                                            self.config_directory)
        [dataset, data] = mls.workflows.tasks.LoadDataTask.load_data(config, log)
        self.assertIsInstance(dataset, mls.sl.datasets.NClassRandomClassificationWithNoise)
        self.assertIsInstance(data, mls.sl.models.DataPandas)
        self.assertEqual(100, len(data.x))
        self.assertEqual(100, len(data.y))
        self.assertEqual(3, len(data.df.keys().tolist()))

    def test_load_data_keep_columns(self):
        """
        :test : mlsurvey.workflows.tasks.LoadDataTask.load_data()
        :condition : config contains parameter to keep some columns
        :main_result : load the data with some columns only
        """
        config, log = self._init_config_log('config_filedataset_keep_columns.json',
                                            self.base_directory,
                                            self.config_directory)
        [_, data] = mls.workflows.tasks.LoadDataTask.load_data(config, log, self.base_directory)
        self.assertListEqual(['Column1', 'Column2 '], data.df.keys().tolist())

    def test_get_node(self):
        """
        :test : mlsurvey.workflows.tasks.LoadDataTask.get_node()
        :condition : -
        :main_result : create a kedro with input and output parameter
        """
        load_data_node = mls.workflows.tasks.LoadDataTask.get_node()
        self.assertIsInstance(load_data_node, Node)

    def _run_one_task(self, config_filename):
        # create node from Task
        load_data_node = mls.workflows.tasks.LoadDataTask.get_node()
        config, log = self._init_config_log(config_filename, self.base_directory, self.config_directory)
        # Prepare a data catalog
        data_catalog = DataCatalog({'config': MemoryDataSet(),
                                    'log': MemoryDataSet(),
                                    'base_directory': MemoryDataSet(),
                                    'dataset': MemoryDataSet(),
                                    'raw_data': MemoryDataSet()})
        data_catalog.save('config', config)
        data_catalog.save('log', log)
        data_catalog.save('base_directory', self.base_directory)
        # Assemble nodes into a pipeline
        pipeline = Pipeline([load_data_node])
        # Create a runner to run the pipeline
        runner = SequentialRunner()
        # Run the pipeline
        runner.run(pipeline, data_catalog)
        return log, config, data_catalog

    def test_run(self):
        """
        :test : mlsurvey.workflows.tasks.LoadDataTask.run()
        :condition : -
        :main_result : data file are loaded, saved in hdf database and logged, config is logged into mlflow
        """
        config_filename = 'complete_config_loaded.json'
        log, config, data_catalog = self._run_one_task(config_filename)
        dataset = data_catalog.load('dataset')
        data = data_catalog.load('raw_data')

        self.assertIsInstance(dataset, mls.sl.datasets.NClassRandomClassificationWithNoise)
        self.assertIsInstance(data, mls.sl.models.DataPandas)
        self.assertEqual(100, len(data.x))
        self.assertEqual(100, len(data.y))

        self.assertTrue(os.path.isfile(os.path.join(log.directory,  'config.json')))
        self.assertEqual('231279b44f04b545748cf614a34ad3d4',
                         mls.Utils.md5_file(os.path.join(log.directory, 'config.json')))
        log.set_sub_dir(str(mls.workflows.tasks.LoadDataTask.__name__))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'dataset.json')))
        self.assertEqual('8c61e35b282706649f63fba48d3d103e',
                         mls.Utils.md5_file(os.path.join(log.directory, 'dataset.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'raw_data.json')))
        self.assertEqual('06d2fbdd897691e283a6d08559ae99c5',
                         mls.Utils.md5_file(os.path.join(log.directory, 'raw_data.json')))
        df_raw_data = mls.FileOperation.read_hdf('raw_data-content.h5',
                                                 log.directory, 'Pandas')
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data)
        self.assertEqual(100, len(raw_data.x))
        self.assertEqual(100, len(raw_data.y))
        # parameters in mlflow
        self.assertIn('input.type', log.mlflow_client.get_run(log.mlflow_run.info.run_id).data.params)

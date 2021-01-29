import unittest
import os
import shutil

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.runner import SequentialRunner
import mlflow

import mlsurvey as mls


class TestLoadDataTask(unittest.TestCase):
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

        dataset = mls.sl.workflows.tasks.LoadDataTask.init_dataset(ds_dict)
        self.assertIsInstance(dataset, mls.sl.datasets.NClassRandomClassificationWithNoise)
        self.assertDictEqual(dataset.params, ds_param_dict)
        self.assertDictEqual({}, dataset.fairness)

    def test_load_data(self):
        """
        :test : mlsurvey.workflows.tasks.LoadDataTask.load_data()
        :condition : -
        :main_result : load the data
        """
        # TODO revoir gestion configuation et logging
        final_config_directory = os.path.join(str(self.base_directory), str(self.config_directory))
        config = mls.Config(name='complete_config_loaded.json', directory=final_config_directory)
        config.compact()
        temp_log = mls.Logging()
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir),
                          dir_name=temp_log.dir_name)

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
        # TODO revoir gestion configuation et logging
        final_config_directory = os.path.join(str(self.base_directory), str(self.config_directory))
        config = mls.Config(name='config_filedataset_keep_columns.json', directory=final_config_directory)
        config.compact()
        temp_log = mls.Logging()
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir),
                          dir_name=temp_log.dir_name)

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

    def test_run(self):
        """
        :test : mlsurvey.workflows.tasks.LoadDataTask.run()
        :condition : -
        :main_result : data file are loaded, saved in hdf database and logged, config is logged into mlflow
        """
        # create node from Task
        load_data_node = mls.workflows.tasks.LoadDataTask.get_node()
        # Prepare a data catalog
        # TODO revoir gestion configuation et Logging
        final_config_directory = os.path.join(str(self.base_directory), str(self.config_directory))
        config = mls.Config(name='complete_config_loaded.json', directory=final_config_directory)
        config.compact()
        # init logging
        log = mls.Logging(base_dir=os.path.join(self.base_directory, 'logs'),
                          mlflow_log=True)

        data_catalog = DataCatalog({'config': MemoryDataSet(),
                                    'log': MemoryDataSet(),
                                    'dataset': MemoryDataSet(),
                                    'data': MemoryDataSet()})
        data_catalog.save('config', config)
        data_catalog.save('log', log)
        # Assemble nodes into a pipeline
        pipeline = Pipeline([load_data_node])
        # Create a runner to run the pipeline
        runner = SequentialRunner()
        # Run the pipeline
        runner.run(pipeline, data_catalog)
        dataset = data_catalog.load('dataset')
        data = data_catalog.load('data')

        self.assertIsInstance(dataset, mls.sl.datasets.NClassRandomClassificationWithNoise)
        self.assertIsInstance(data, mls.sl.models.DataPandas)
        self.assertEqual(100, len(data.x))
        self.assertEqual(100, len(data.y))

        self.assertTrue(os.path.isfile(os.path.join(log.directory,
                                                    str(mls.workflows.tasks.LoadDataTask.__name__),
                                                    'config.json')))
        self.assertEqual('231279b44f04b545748cf614a34ad3d4',
                         mls.Utils.md5_file(os.path.join(log.directory,
                                                         str(mls.workflows.tasks.LoadDataTask.__name__),
                                                         'config.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory,
                                                    str(mls.workflows.tasks.LoadDataTask.__name__),
                                                    'dataset.json')))
        self.assertEqual('8c61e35b282706649f63fba48d3d103e',
                         mls.Utils.md5_file(os.path.join(log.directory,
                                                         str(mls.workflows.tasks.LoadDataTask.__name__),
                                                         'dataset.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory,
                                                    str(mls.workflows.tasks.LoadDataTask.__name__),
                                                    'raw_data.json')))
        self.assertEqual('06d2fbdd897691e283a6d08559ae99c5',
                         mls.Utils.md5_file(os.path.join(log.directory,
                                                         str(mls.workflows.tasks.LoadDataTask.__name__),
                                                         'raw_data.json')))
        df_raw_data = mls.FileOperation.read_hdf('raw_data-content.h5',
                                                 os.path.join(log.base_dir,
                                                              log.dir_name,
                                                              str(mls.workflows.tasks.LoadDataTask.__name__)),
                                                 'Pandas')
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data)
        self.assertEqual(100, len(raw_data.x))
        self.assertEqual(100, len(raw_data.y))
        # parameters in mlflow
        self.assertIn('input.type', log.mlflow_client.get_run(log.mlflow_run.info.run_id).data.params)

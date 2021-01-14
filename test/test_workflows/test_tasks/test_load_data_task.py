import os
import shutil
import unittest

import mlflow

import mlsurvey as mls


class TestLoadDataTask(unittest.TestCase):
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

    def test_init_dataset(self):
        """
        :test : mlsurvey.workflows.tasks.LoadDataTask.init_dataset()
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

    def test_run(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LoadDataTask.run()
        :condition : -
        :main_result : data file are loaded, saved in hdf database and logged, config is logged into mlflow
        """
        temp_log = mls.Logging()
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        task = mls.sl.workflows.tasks.LoadDataTask(logging_directory=temp_log.dir_name,
                                                   logging_base_directory=os.path.join(self.base_directory,
                                                                                       temp_log.base_dir),
                                                   config_filename='complete_config_loaded.json',
                                                   config_directory=self.config_directory,
                                                   base_directory=self.base_directory,
                                                   mlflow_run_id=run.info.run_id)
        node = task.def_node()
        node.run(dict(input=2, output=3))

        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir),
                          dir_name=temp_log.dir_name,
                          mlflow_run_id=run.info.run_id)
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'config.json')))
        self.assertEqual('231279b44f04b545748cf614a34ad3d4',
                         mls.Utils.md5_file(os.path.join(log.directory, 'config.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'dataset.json')))
        self.assertEqual('8c61e35b282706649f63fba48d3d103e',
                         mls.Utils.md5_file(os.path.join(log.directory, 'dataset.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'raw_data.json')))
        self.assertEqual('06d2fbdd897691e283a6d08559ae99c5',
                         mls.Utils.md5_file(os.path.join(log.directory, 'raw_data.json')))
        df_raw_data = mls.FileOperation.read_hdf('raw_data-content.h5',
                                                 os.path.join(log.base_dir, log.dir_name),
                                                 'Pandas')
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data)
        self.assertEqual(100, len(raw_data.x))
        self.assertEqual(100, len(raw_data.y))
        # parameters in mlflow
        self.assertIn('input.type', log.mlflow_client.get_run(log.mlflow_run_id).data.params)

    def test_run_filedataset_data_is_obtained_pandas_implicit(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LoadDataTask.run()
        :condition : pandas is implicit in the config file
        :main_result : dataset, raw_data properties are ok
        """
        temp_log = mls.Logging()
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        luigi.build([mls.sl.workflows.tasks.LoadDataTask(logging_directory=temp_log.dir_name,
                                                         logging_base_directory=os.path.join(self.base_directory,
                                                                                             temp_log.base_dir),
                                                         config_filename='config_filedataset.json',
                                                         config_directory=self.config_directory,
                                                         base_directory=self.base_directory,
                                                         mlflow_run_id=run.info.run_id)], local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        df_raw_data = mls.FileOperation.read_hdf('raw_data-content.h5',
                                                 os.path.join(log.base_dir, log.dir_name),
                                                 'Pandas')

        dataset_dict = log.load_json_as_dict(os.path.join('dataset.json'))
        dataset = mls.sl.datasets.DataSetFactory.create_dataset_from_dict(dataset_dict)
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data,
                                                         y_col_name=dataset.metadata['y_col_name'])
        self.assertDictEqual({'protected_attribute': 1,
                              'privileged_classes': 'x >= 25',
                              'target_is_one': 'good',
                              'target_is_zero': 'bad'}, dataset.fairness)
        self.assertEqual(13, len(raw_data.x))
        self.assertEqual(13, len(raw_data.y))
        self.assertListEqual(['credit_amount', 'age', 'class'], raw_data.df.keys().to_list())

    def test_run_filedataset_data_is_obtained_pandas_explicit(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LoadDataTask.run()
        :condition : pandas is explicit in the config file
        :main_result : data file are loaded, saved in hdf database and logged
        """
        temp_log = mls.Logging()
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        luigi.build([mls.sl.workflows.tasks.LoadDataTask(logging_directory=temp_log.dir_name,
                                                         logging_base_directory=os.path.join(self.base_directory,
                                                                                             temp_log.base_dir),
                                                         config_filename='config_filedataset_pandas_explicit.json',
                                                         config_directory=self.config_directory,
                                                         base_directory=self.base_directory,
                                                         mlflow_run_id=run.info.run_id)], local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        dataset_dict = log.load_json_as_dict(os.path.join('dataset.json'))
        dataset = mls.sl.datasets.DataSetFactory.create_dataset_from_dict(dataset_dict)
        df_raw_data = mls.FileOperation.read_hdf('raw_data-content.h5',
                                                 os.path.join(log.base_dir, log.dir_name),
                                                 'Pandas')
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data,
                                                         y_col_name=dataset.metadata['y_col_name'])
        self.assertDictEqual({'protected_attribute': 1,
                              'privileged_classes': 'x >= 25',
                              'target_is_one': 'good',
                              'target_is_zero': 'bad'}, dataset.fairness)
        self.assertEqual(13, len(raw_data.x))
        self.assertEqual(13, len(raw_data.y))

    def test_run_fairness_data_is_obtained_with_config(self):
        """
        :test : mls.sl.workflows.tasks.LoadDataTask.run()
        :condition : config file contains fairness parameters
        :main_result : Generate and dataset contains the fairness parameters
        """
        temp_log = mls.Logging()
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        luigi.build([mls.sl.workflows.tasks.LoadDataTask(logging_directory=temp_log.dir_name,
                                                         logging_base_directory=os.path.join(self.base_directory,
                                                                                             temp_log.base_dir),
                                                         config_filename='config_filedataset.json',
                                                         config_directory=self.config_directory,
                                                         base_directory=self.base_directory,
                                                         mlflow_run_id=run.info.run_id)], local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        dataset_dict = log.load_json_as_dict(os.path.join('dataset.json'))
        dataset = mls.sl.datasets.DataSetFactory.create_dataset_from_dict(dataset_dict)
        self.assertIsInstance(dataset, mls.sl.datasets.FileDataSet)
        self.assertEqual(1, dataset.fairness['protected_attribute'])
        self.assertEqual("x >= 25", dataset.fairness['privileged_classes'])

    def test_run_keep_certains_columns(self):
        """
        :test : mls.sl.workflows.tasks.LoadDataTask.run()
        :condition : config file contains keeping columns
        :main_result : the result loaded data contains only kept columns
        """
        temp_log = mls.Logging()
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        luigi.build([mls.sl.workflows.tasks.LoadDataTask(logging_directory=temp_log.dir_name,
                                                         logging_base_directory=os.path.join(self.base_directory,
                                                                                             temp_log.base_dir),
                                                         config_filename='config_filedataset_keep_columns.json',
                                                         config_directory=self.config_directory,
                                                         base_directory=self.base_directory,
                                                         mlflow_run_id=run.info.run_id)],
                    local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        df_raw_data = mls.FileOperation.read_hdf('raw_data-content.h5',
                                                 os.path.join(log.base_dir, log.dir_name),
                                                 'Pandas')
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data)
        self.assertListEqual(['Column1', 'Column2 '], raw_data.df.keys().tolist())

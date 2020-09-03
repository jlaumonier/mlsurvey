import os
import shutil
import unittest

import luigi

import mlsurvey as mls


class TestLoadDataTask(unittest.TestCase):

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

    def test_run(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LoadDataTask.run()
        :condition : -
        :main_result : data file are loaded, saved in hdf database and logged
        """
        temp_log = mls.Logging()
        luigi.build([mls.sl.workflows.tasks.LoadDataTask(logging_directory=temp_log.dir_name,
                                                         logging_base_directory=os.path.join(self.base_directory,
                                                                                             temp_log.base_dir),
                                                         config_filename='complete_config_loaded.json',
                                                         config_directory=self.config_directory,
                                                         base_directory=self.base_directory)], local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'config.json')))
        self.assertEqual('ae28ce16852d7a5ddbecdb07ea755339',
                         mls.Utils.md5_file(os.path.join(log.directory, 'config.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'dataset.json')))
        self.assertEqual('8c61e35b282706649f63fba48d3d103e',
                         mls.Utils.md5_file(os.path.join(log.directory, 'dataset.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'raw_data.json')))
        self.assertEqual('8f8f7e5c285e42cde7a0da9fd55a31f9',
                         mls.Utils.md5_file(os.path.join(log.directory, 'raw_data.json')))
        df_raw_data = mls.FileOperation.read_hdf('raw_data.h5', os.path.join(log.base_dir, log.dir_name), 'Pandas')
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data)
        self.assertEqual(100, len(raw_data.x))
        self.assertEqual(100, len(raw_data.y))

    def test_run_filedataset_data_is_obtained_pandas_implicit(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LoadDataTask.run()
        :condition : pandas is implicit in the config file
        :main_result : dataset, raw_data properties are ok
        """
        temp_log = mls.Logging()
        luigi.build([mls.sl.workflows.tasks.LoadDataTask(logging_directory=temp_log.dir_name,
                                                         logging_base_directory=os.path.join(self.base_directory,
                                                                                             temp_log.base_dir),
                                                         config_filename='config_filedataset.json',
                                                         config_directory=self.config_directory,
                                                         base_directory=self.base_directory)], local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        df_raw_data = mls.FileOperation.read_hdf('raw_data.h5', os.path.join(log.base_dir, log.dir_name), 'Pandas')
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data)
        dataset_dict = log.load_json_as_dict(os.path.join('dataset.json'))
        dataset = mls.sl.datasets.DataSetFactory.create_dataset_from_dict(dataset_dict)
        self.assertDictEqual({'protected_attribute': 1, 'privileged_classes': 'x >= 25'}, dataset.fairness)
        self.assertEqual(13, len(raw_data.x))
        self.assertEqual(13, len(raw_data.y))

    def test_run_filedataset_data_is_obtained_pandas_explicit(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LoadDataTask.run()
        :condition : pandas is explicit in the config file
        :main_result : data file are loaded, saved in hdf database and logged
        """
        temp_log = mls.Logging()
        luigi.build([mls.sl.workflows.tasks.LoadDataTask(logging_directory=temp_log.dir_name,
                                                         logging_base_directory=os.path.join(self.base_directory,
                                                                                             temp_log.base_dir),
                                                         config_filename='config_filedataset_pandas_explicit.json',
                                                         config_directory=self.config_directory,
                                                         base_directory=self.base_directory)], local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        df_raw_data = mls.FileOperation.read_hdf('raw_data.h5', os.path.join(log.base_dir, log.dir_name), 'Pandas')
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data)
        dataset_dict = log.load_json_as_dict(os.path.join('dataset.json'))
        dataset = mls.sl.datasets.DataSetFactory.create_dataset_from_dict(dataset_dict)
        self.assertDictEqual({'protected_attribute': 1, 'privileged_classes': 'x >= 25'}, dataset.fairness)
        self.assertEqual(13, len(raw_data.x))
        self.assertEqual(13, len(raw_data.y))

    def test_run_fairness_data_is_obtained_with_config(self):
        """
        :test : mls.sl.workflows.tasks.LoadDataTask.run()
        :condition : config file contains fairness parameters
        :main_result : Generate and dataset contains the fairness parameters
        """
        temp_log = mls.Logging()
        luigi.build([mls.sl.workflows.tasks.LoadDataTask(logging_directory=temp_log.dir_name,
                                                         logging_base_directory=os.path.join(self.base_directory,
                                                                                             temp_log.base_dir),
                                                         config_filename='config_filedataset.json',
                                                         config_directory=self.config_directory,
                                                         base_directory=self.base_directory)], local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        dataset_dict = log.load_json_as_dict(os.path.join('dataset.json'))
        dataset = mls.sl.datasets.DataSetFactory.create_dataset_from_dict(dataset_dict)
        self.assertIsInstance(dataset, mls.sl.datasets.FileDataSet)
        self.assertEqual(1, dataset.fairness['protected_attribute'])
        self.assertEqual("x >= 25", dataset.fairness['privileged_classes'])


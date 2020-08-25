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
        shutil.rmtree(log.base_dir, ignore_errors=True)

    def test_init_log_config(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LoadDataTask.init_log_config()
        :condition : -
        :main_result : Log and Config objects are initialized
        """
        log = mls.Logging()
        task = mls.sl.workflows.tasks.LoadDataTask(logging_base_directory=log.base_dir,
                                                   logging_directory=log.dir_name,
                                                   config_filename='complete_config_loaded.json',
                                                   config_directory=self.config_directory)
        task.init_log_config()
        self.assertIsInstance(task.config, mls.Config)
        self.assertIsNotNone(task.config.data)
        self.assertIsInstance(task.log, mls.Logging)

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
        log = mls.Logging()
        luigi.build([mls.sl.workflows.tasks.LoadDataTask(logging_directory=log.dir_name,
                                                         logging_base_directory=log.base_dir,
                                                         config_filename='complete_config_loaded.json',
                                                         config_directory=self.config_directory)], local_scheduler=True)
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'config.json')))
        self.assertEqual('ae28ce16852d7a5ddbecdb07ea755339',
                         mls.Utils.md5_file(os.path.join(log.directory, 'config.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'dataset.json')))
        self.assertEqual('8c61e35b282706649f63fba48d3d103e',
                         mls.Utils.md5_file(os.path.join(log.directory, 'dataset.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'raw_data.json')))
        self.assertEqual('c3406a4ccbbd3bbf2fa0fbd03f167f68',
                         mls.Utils.md5_file(os.path.join(log.directory, 'raw_data.json')))

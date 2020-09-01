import os
import shutil
import unittest

import luigi

import mlsurvey as mls


class TestSplitDataTask(unittest.TestCase):
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

    def test_run(self):
        """
        :test : mlsurvey.sl.workflows.tasks.SplitDataTask.run()
        :condition : Data are prepared, saved in hdf database and logged
        :main_result : data are split (train test).
        """
        log = mls.Logging()
        luigi.build([mls.sl.workflows.tasks.SplitDataTask(logging_directory=log.dir_name,
                                                          logging_base_directory=log.base_dir,
                                                          config_filename='complete_config_loaded.json',
                                                          config_directory=self.config_directory)],
                    local_scheduler=True)
        df_data = mls.FileOperation.read_hdf('data.h5', os.path.join(log.base_dir, log.dir_name), 'Pandas')
        data = mls.sl.models.DataFactory.create_data('Pandas', df_data)
        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'train.h5')))
        df_train = mls.FileOperation.read_hdf('train.h5', os.path.join(log.base_dir, log.dir_name), 'Pandas')
        data_train = mls.sl.models.DataFactory.create_data('Pandas', df_train)
        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'test.h5')))
        df_test = mls.FileOperation.read_hdf('test.h5', os.path.join(log.base_dir, log.dir_name), 'Pandas')
        data_test = mls.sl.models.DataFactory.create_data('Pandas', df_test)
        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'test.h5')))
        self.assertEqual(100, len(data.x))
        self.assertEqual(100, len(data.y))
        self.assertEqual(20, len(data_test.x))
        self.assertEqual(20, len(data_test.y))
        self.assertEqual(80, len(data_train.x))
        self.assertEqual(80, len(data_train.y))

import os
import unittest
import shutil

import luigi

import mlsurvey as mls


class TestPrepareDataTask(unittest.TestCase):

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

    def test_run(self):
        """
        :test : mlsurvey.sl.workflows.tasks.PrepareDataTask.run()
        :condition : data file are loaded, saved in hdf database and logged
        :main_result : data are prepared.
        """
        temp_log = mls.Logging()
        luigi.build([mls.sl.workflows.tasks.PrepareDataTask(logging_directory=temp_log.dir_name,
                                                            logging_base_directory=os.path.join(self.base_directory,
                                                                                                temp_log.base_dir),
                                                            config_filename='complete_config_loaded.json',
                                                            config_directory=self.config_directory)],
                    local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        df_raw_data = mls.FileOperation.read_hdf('raw_data-content.h5',
                                                 os.path.join(log.base_dir, log.dir_name),
                                                 'Pandas')
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data)
        lx = len(raw_data.x)
        ly = len(raw_data.y)
        self.assertEqual(-1.766054694735782, raw_data.x[0][0])
        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'data-content.h5')))
        df_data = mls.FileOperation.read_hdf('data-content.h5', os.path.join(log.base_dir, log.dir_name), 'Pandas')
        data = mls.sl.models.DataFactory.create_data('Pandas', df_data)
        self.assertEqual(-0.7655005998158294, data.x[0][0])
        self.assertEqual(lx, len(data.x))
        self.assertEqual(ly, len(data.y))

    def test_run_prepare_textual_data(self):
        """
        :test : mlsurvey.sl.workflows.tasks.PrepareDataTask.run()
        :condition : data is textual
        :main_result : data are prepared.
        """
        temp_log = mls.Logging()
        luigi.build([mls.sl.workflows.tasks.PrepareDataTask(logging_directory=temp_log.dir_name,
                                                            logging_base_directory=os.path.join(self.base_directory,
                                                                                                temp_log.base_dir),
                                                            config_filename='config_dataset_text.json',
                                                            config_directory=self.config_directory)],
                    local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        df_raw_data = mls.FileOperation.read_hdf('raw_data-content.h5',
                                                 os.path.join(log.base_dir, log.dir_name),
                                                 'Pandas')
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data)
        lx = len(raw_data.x)
        ly = len(raw_data.y)
        self.assertEqual('7', raw_data.y[0])
        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'data-content.h5')))
        df_data = mls.FileOperation.read_hdf('data-content.h5', os.path.join(log.base_dir, log.dir_name), 'Pandas')
        data = mls.sl.models.DataFactory.create_data('Pandas', df_data)
        self.assertEqual(0.23989072176612425, data.x[0][0])
        self.assertEqual(lx, len(data.x))
        self.assertEqual(ly, len(data.y))

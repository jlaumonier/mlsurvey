import os
import unittest
import shutil

import luigi

import mlsurvey as mls


class TestPrepareDataTask(unittest.TestCase):

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
        :test : mlsurvey.sl.workflows.tasks.PrepareDataTask.run()
        :condition : data file are loaded, saved in hdf database and logged
        :main_result : data are prepared.
        """
        log = mls.Logging()
        luigi.build([mls.sl.workflows.tasks.PrepareDataTask(logging_directory=log.dir_name,
                                                            logging_base_directory=log.base_dir,
                                                            config_filename='complete_config_loaded.json',
                                                            config_directory=self.config_directory)],
                    local_scheduler=True)
        df_raw_data = mls.FileOperation.read_hdf('raw_data.h5', os.path.join(log.base_dir, log.dir_name), 'Pandas')
        raw_data = mls.sl.models.DataFactory.create_data('Pandas', df_raw_data)
        lx = len(raw_data.x)
        ly = len(raw_data.y)
        self.assertEqual(-1.766054694735782, raw_data.x[0][0])
        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'data.h5')))
        df_data = mls.FileOperation.read_hdf('data.h5', os.path.join(log.base_dir, log.dir_name), 'Pandas')
        data = mls.sl.models.DataFactory.create_data('Pandas', df_data)
        self.assertEqual(-0.7655005998158294, data.x[0][0])
        self.assertEqual(lx, len(data.x))
        self.assertEqual(ly, len(data.y))

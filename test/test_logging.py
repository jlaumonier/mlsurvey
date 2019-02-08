import datetime
import os
import unittest

import mlsurvey as mls


class TestLogging(unittest.TestCase):

    def test_init_log_directory_created_with_date(self):
        log = mls.Logging()
        dh = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "/"
        self.assertTrue(os.path.isdir(log.base_dir + dh))
        self.assertEqual(log.directory, log.base_dir + dh)

    def test_init_log_directory_create_with_fixed_name(self):
        dir_name = 'testing/'
        _ = mls.Logging(dir_name=dir_name)
        self.assertTrue(os.path.isdir('logs/' + dir_name))

    def test_save_input_input_saved(self):
        dir_name = 'testing/'
        d = mls.datasets.DataSetFactory.create_dataset("Iris")
        d.generate()
        i = mls.Input()
        i.set_data(d)
        log = mls.Logging(dir_name)
        log.save_input(i)
        self.assertTrue(os.path.isfile(log.directory + 'input.json'))
        f = open(log.directory + 'input.json', 'r')
        contents = f.read()
        self.assertEqual('{"input.x": [[5.1, 3.5], ', contents[:25])
        f.close()

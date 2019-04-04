import os
import unittest

import mlsurvey as mls


class TestAnalyzeLogs(unittest.TestCase):
    directory = ''

    @classmethod
    def setUpClass(cls):
        d = os.path.dirname(__file__)
        cls.directory = os.path.join(d, '../files/visualize-log//')

    def test_init_should_init(self):
        analyse_logs = mls.visualize.AnalyzeLogs(self.directory)
        expected_list_dir = ['directory1', 'directory2', 'directory3']
        expected_list_full_dir = [os.path.join(self.directory, 'directory1'),
                                  os.path.join(self.directory, 'directory2'),
                                  os.path.join(self.directory, 'directory3')]
        self.assertEqual(analyse_logs.directory, self.directory)
        self.assertEqual(len(analyse_logs.list_dir), len(analyse_logs.list_full_dir))
        self.assertListEqual(analyse_logs.list_dir, expected_list_dir)
        self.assertListEqual(analyse_logs.list_full_dir, expected_list_full_dir)

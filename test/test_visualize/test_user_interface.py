import os
import unittest

import mlsurvey as mls


class TestUserInterface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        d = os.path.dirname(__file__)
        cls.directory = os.path.join(d, '../files/visualize-log//')

    def test_init_should_init(self):
        app_interface = mls.visualize.UserInterface(self.directory)
        self.assertIsInstance(app_interface.analyse_logs, mls.visualize.AnalyzeLogs)
        self.assertEqual(len(app_interface.analyse_logs.db.all()), 3)

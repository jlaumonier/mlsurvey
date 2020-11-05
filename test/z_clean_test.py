import unittest
import shutil


class ZCleanTest(unittest.TestCase):
    """ Class used only to clean some temporary directories created by some #!$/ libraries such as mlflow"""

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('mlruns', ignore_errors=True)

    def test_clean(self):
        self.assertTrue(True)

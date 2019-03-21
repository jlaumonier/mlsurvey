import unittest

import mlsurvey as mls
import mlsurvey.datasets as datasets


class TestCompleteImport(unittest.TestCase):

    def test_import(self):
        var = datasets.DataSet('type')
        self.assertIsInstance(var, mls.datasets.DataSet)

    def test_import_complete(self):
        var = mls.datasets.DataSet('type')
        self.assertIsInstance(var, datasets.DataSet)

    def test_import_error(self):
        try:
            _ = mls.DataSet('type')
            self.assertTrue(False)
        except AttributeError:
            self.assertTrue(True)

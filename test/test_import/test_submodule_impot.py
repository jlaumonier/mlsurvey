import unittest

import mlsurvey
import mlsurvey as mls
import mlsurvey.sl.datasets as datasets


class TestCompleteImport(unittest.TestCase):

    def test_import(self):
        var = datasets.DataSet('type')
        self.assertIsInstance(var, mls.sl.datasets.DataSet)

    def test_import_complete(self):
        var = mlsurvey.sl.datasets.DataSet('type')
        self.assertIsInstance(var, datasets.DataSet)

    def test_import_error(self):
        try:
            # This line makes a unresolved reference warning but that's what i want to test the packaging
            _ = mls.sl.DataSet('type')
            self.assertTrue(False)
        except AttributeError:
            self.assertTrue(True)

import unittest

import mlsurvey as mls


class TestSimpleImport(unittest.TestCase):

    def test_import(self):
        c = {'c': 'test'}
        var = mls.Config(config=c)
        self.assertIsInstance(var, mls.Config)

    def test_import_complete(self):
        c = {'c': 'test'}
        var = mls.config.Config(config=c)
        self.assertIsInstance(var, mls.Config)

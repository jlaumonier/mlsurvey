import unittest

import mlsurvey.config as config


class TestExplicitImport(unittest.TestCase):

    def test_import(self):
        c = {'c': 'test'}
        var = config.Config(config=c)
        self.assertIsInstance(var, config.Config)

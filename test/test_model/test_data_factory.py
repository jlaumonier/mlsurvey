import unittest

import numpy as np
import pandas as pd

import mlsurvey as mls


class TestDataFactory(unittest.TestCase):
    fac = {}

    @classmethod
    def setUpClass(cls):
        cls.fac = dict(mls.models.DataFactory.factories)

    @classmethod
    def tearDownClass(cls):
        mls.models.DataFactory.factories = dict(cls.fac)

    def setUp(self):
        mls.models.DataFactory().factories.clear()

    def test_init_dataset_factory_should_be_initialized(self):
        data_factory = mls.models.DataFactory()
        self.assertIsNotNone(data_factory)
        self.assertDictEqual({}, data_factory.factories)

    def test_add_factory_should_be_added(self):
        mls.models.DataFactory.add_factory('Pandas',
                                           mls.models.DataPandas.Factory)
        self.assertEqual(1, len(mls.models.DataFactory.factories))

    def test_create_data_should_be_generated(self):
        data_factory = mls.models.DataFactory()
        mls.models.DataFactory.add_factory('Pandas',
                                           mls.models.DataPandas.Factory)
        x = np.array([[1, 2], [3, 4]])
        df = pd.DataFrame(x)
        data = data_factory.create_data('Pandas', df)
        self.assertIsInstance(data, mls.models.DataPandas)

    def test_create_data_from_dict_created(self):
        source = {
            "df_contains": "xy",
            "y_col_name": "target",
            "y_pred_col_name": "target_pred"
        }
        data_factory = mls.models.DataFactory()
        mls.models.DataFactory.add_factory('Pandas',
                                           mls.models.DataPandas.Factory)
        x = np.array([[1, 2], [3, 4]])
        df = pd.DataFrame(x)
        data = data_factory.create_data_from_dict('Pandas', source, df)
        self.assertIsInstance(data, mls.models.DataPandas)
        self.assertEqual('xy', data.df_contains)
        self.assertEqual('target', data.y_col_name)
        self.assertEqual('target_pred', data.y_pred_col_name)

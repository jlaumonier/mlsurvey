import unittest

import mlsurvey as mls


class TestDataSetFactory(unittest.TestCase):
    fac = {}

    @classmethod
    def setUpClass(cls):
        cls.fac = dict(mls.sl.datasets.DataSetFactory.factories)

    @classmethod
    def tearDownClass(cls):
        mls.sl.datasets.DataSetFactory.factories = dict(cls.fac)

    def setUp(self):
        mls.sl.datasets.DataSetFactory().factories.clear()

    def test_init_dataset_factory_should_be_initialized(self):
        dataset_factory = mls.sl.datasets.DataSetFactory()
        self.assertIsNotNone(dataset_factory)
        self.assertDictEqual({}, dataset_factory.factories)

    def test_add_factory_should_be_added(self):
        mls.sl.datasets.DataSetFactory.add_factory('NClassRandomClassificationWithNoise',
                                                   mls.sl.datasets.NClassRandomClassificationWithNoise.Factory)
        self.assertEqual(1, len(mls.sl.datasets.DataSetFactory.factories))

    def test_create_dataset_should_be_generated(self):
        dataset_factory = mls.sl.datasets.DataSetFactory()
        mls.sl.datasets.DataSetFactory.add_factory('NClassRandomClassificationWithNoise',
                                                   mls.sl.datasets.NClassRandomClassificationWithNoise.Factory)
        data = dataset_factory.create_dataset('NClassRandomClassificationWithNoise')
        df = data.generate()
        self.assertEqual('NClassRandomClassificationWithNoise', data.t)
        self.assertEqual(100, df.iloc[:, 0:-1].shape[0])
        self.assertEqual(2, df.iloc[:, 0:-1].shape[1])
        self.assertEqual(100, df.iloc[:, -1].shape[0])

    def test_create_dataset_factory_not_exists_generic_created(self):
        dataset_factory = mls.sl.datasets.DataSetFactory()
        mls.sl.datasets.DataSetFactory.add_factory('NClassRandomClassificationWithNoise',
                                                   mls.sl.datasets.NClassRandomClassificationWithNoise.Factory)
        mls.sl.datasets.DataSetFactory.add_factory('generic',
                                                   mls.sl.datasets.GenericDataSet.Factory)
        data = dataset_factory.create_dataset('NotExistedFactory')
        self.assertIsInstance(data, mls.sl.datasets.GenericDataSet)
        self.assertEqual('NotExistedFactory', data.t)

    def test_create_dataset_from_dict_created(self):
        """
        :test : mlsurvey.sl.dataset.DatasetFactory.create_dataset_from_dict()
        :condition : config contains type, storage, parameters, metadata, fairness
        :main_result : dataset created with correct parameters
        """
        params = {'param1': 1, 'param2': 3, 'return_X_y': False}
        metadata = {"y_col_name": "one_target"}
        fairness = {"protected_attribute": 12,
                    "privileged_classes": "x >= 25",
                    "target_is_one": 1,
                    "target_is_zero": 0
                    }
        source = {'type': 'load_iris',
                  'storage': 'Pandas',
                  'parameters': params,
                  "metadata": metadata,
                  'fairness': fairness
                  }
        dataset_factory = mls.sl.datasets.DataSetFactory()
        mls.sl.datasets.DataSetFactory.add_factory('generic',
                                                   mls.sl.datasets.GenericDataSet.Factory)

        dataset = dataset_factory.create_dataset_from_dict(source)
        self.assertEqual(dataset.storage, 'Pandas')
        self.assertEqual('load_iris', dataset.t)
        self.assertDictEqual(params, dataset.params)
        self.assertDictEqual(fairness, dataset.fairness)
        self.assertDictEqual(metadata, dataset.metadata)

    def test_create_dataset_from_dict_created_no_optional_params(self):
        """
        :test : mlsurvey.sl.dataset.DatasetFactory.create_dataset_from_dict()
        :condition : config contains type, parameters, no storage,  no metadata, no fairness
        :main_result : dataset created with correct parameters
        """
        params = {'param1': 1, 'param2': 3, 'return_X_y': False}
        metadata = {"y_col_name": "target"}
        source = {'type': 'load_iris',
                  'parameters': params
                  }
        dataset_factory = mls.sl.datasets.DataSetFactory()
        mls.sl.datasets.DataSetFactory.add_factory('generic',
                                                   mls.sl.datasets.GenericDataSet.Factory)

        dataset = dataset_factory.create_dataset_from_dict(source)
        self.assertEqual(dataset.storage, 'Pandas')
        self.assertEqual('load_iris', dataset.t)
        self.assertDictEqual(params, dataset.params)
        self.assertDictEqual({}, dataset.fairness)
        self.assertDictEqual(metadata, dataset.metadata)

import unittest

import mlsurvey as mls


class TestDataSetFactory(unittest.TestCase):
    fac = {}

    @classmethod
    def setUpClass(cls):
        cls.fac = dict(mls.datasets.DataSetFactory.factories)

    @classmethod
    def tearDownClass(cls):
        mls.datasets.DataSetFactory.factories = dict(cls.fac)

    def setUp(self):
        mls.datasets.DataSetFactory().factories.clear()

    def test_init_dataset_factory_should_be_initialized(self):
        dataset_factory = mls.datasets.DataSetFactory()
        self.assertIsNotNone(dataset_factory)
        self.assertDictEqual({}, dataset_factory.factories)

    def test_add_factory_should_be_added(self):
        mls.datasets.DataSetFactory.add_factory('NClassRandomClassificationWithNoise',
                                                mls.datasets.NClassRandomClassificationWithNoise.Factory)
        self.assertEqual(1, len(mls.datasets.DataSetFactory.factories))

    def test_create_dataset_should_be_generated(self):
        dataset_factory = mls.datasets.DataSetFactory()
        mls.datasets.DataSetFactory.add_factory('NClassRandomClassificationWithNoise',
                                                mls.datasets.NClassRandomClassificationWithNoise.Factory)
        data = dataset_factory.create_dataset('NClassRandomClassificationWithNoise')
        df = data.generate()
        self.assertEqual('NClassRandomClassificationWithNoise', data.t)
        self.assertEqual(100, df.iloc[:, 0:-1].shape[0])
        self.assertEqual(2, df.iloc[:, 0:-1].shape[1])
        self.assertEqual(100, df.iloc[:, -1].shape[0])

    def test_create_dataset_factory_not_exists_generic_created(self):
        dataset_factory = mls.datasets.DataSetFactory()
        mls.datasets.DataSetFactory.add_factory('NClassRandomClassificationWithNoise',
                                                mls.datasets.NClassRandomClassificationWithNoise.Factory)
        mls.datasets.DataSetFactory.add_factory('generic',
                                                mls.datasets.GenericDataSet.Factory)
        data = dataset_factory.create_dataset('NotExistedFactory')
        self.assertIsInstance(data, mls.datasets.GenericDataSet)
        self.assertEqual('NotExistedFactory', data.t)

    def test_create_dataset_from_dict_created(self):
        source = {'type': 'load_iris',
                  'storage': 'Pandas',
                  'parameters': {'param1': 1, 'param2': 3, 'return_X_y': False}}
        dataset_factory = mls.datasets.DataSetFactory()
        mls.datasets.DataSetFactory.add_factory('generic',
                                                mls.datasets.GenericDataSet.Factory)
        params = {'param1': 1, 'param2': 3, 'return_X_y': False}
        dataset = dataset_factory.create_dataset_from_dict(source)
        self.assertEqual(dataset.storage, 'Pandas')
        self.assertEqual('load_iris', dataset.t)
        self.assertDictEqual(params, dataset.params)

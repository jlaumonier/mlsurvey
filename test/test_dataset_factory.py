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
        data.generate()
        self.assertEqual('NClassRandomClassificationWithNoise', data.t)
        self.assertEqual(100, data.x.shape[0])
        self.assertEqual(2, data.x.shape[1])
        self.assertEqual(100, data.y.shape[0])

    def test_create_dataset_factory_not_exists_generic_created(self):
        dataset_factory = mls.datasets.DataSetFactory()
        mls.datasets.DataSetFactory.add_factory('NClassRandomClassificationWithNoise',
                                                mls.datasets.NClassRandomClassificationWithNoise.Factory)
        mls.datasets.DataSetFactory.add_factory('generic',
                                                mls.datasets.GenericDataSet.Factory)
        data = dataset_factory.create_dataset('NotExistedFactory')
        self.assertIsInstance(data, mls.datasets.GenericDataSet)
        self.assertEqual('NotExistedFactory', data.t)

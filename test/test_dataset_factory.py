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
        mls.datasets.DataSetFactory.add_factory('NClassRandomClassification',
                                                mls.datasets.NClassRandomClassification.Factory)
        self.assertEqual(1, len(mls.datasets.DataSetFactory.factories))

    def test_create_dataset_should_be_generated(self):
        dataset_factory = mls.datasets.DataSetFactory()
        mls.datasets.DataSetFactory.add_factory('NClassRandomClassification',
                                                mls.datasets.NClassRandomClassification.Factory)
        data = dataset_factory.create_dataset('NClassRandomClassification')
        data.generate()
        self.assertEqual(100, data.x.shape[0])
        self.assertEqual(2, data.x.shape[1])
        self.assertEqual(100, data.y.shape[0])

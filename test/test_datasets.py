import unittest

import mlsurvey as mls


class TestNClassRandomDataSet(unittest.TestCase):

    def test_init_n_class_random_data_set(self):
        data = mls.datasets.NClassRandomClassification()
        self.assertIsNotNone(data)
        self.assertListEqual([], data.x)
        self.assertIsNotNone([], data.y)

    def test_generate_n_class_random_data_set_is_generated_with_default_params(self):
        data = mls.datasets.NClassRandomClassification()
        data.generate()
        self.assertEqual(100, data.x.shape[0])
        self.assertEqual(20, data.x.shape[1])
        self.assertEqual(100, data.y.shape[0])

    def test_generate_n_class_random_data_set_is_generated_by_factory(self):
        data = mls.datasets.DataSetFactory.create_dataset('NClassRandomClassification')
        data.generate()
        self.assertEqual(100, data.x.shape[0])
        self.assertEqual(20, data.x.shape[1])
        self.assertEqual(100, data.y.shape[0])


class TestIrisDataSet(unittest.TestCase):

    def test_init_iris_data_set(self):
        data = mls.datasets.Iris()
        self.assertIsNotNone(data)
        self.assertListEqual([], data.x)
        self.assertIsNotNone([], data.y)

    def test_generate_iris_data_set_is_generated_with_default_params(self):
        data = mls.datasets.Iris()
        data.generate()
        self.assertEqual(5.1, data.x[0, 0])
        self.assertEqual(0, data.y[0])
        self.assertEqual(2, data.x.shape[1])
        self.assertEqual(150, data.x.shape[0])
        self.assertEqual(150, data.y.shape[0])

    def test_generate_iris_data_set_is_generated_by_factory(self):
        data = mls.datasets.DataSetFactory.create_dataset('Iris')
        data.generate()
        self.assertEqual(5.1, data.x[0, 0])
        self.assertEqual(0, data.y[0])
        self.assertEqual(2, data.x.shape[1])
        self.assertEqual(150, data.x.shape[0])
        self.assertEqual(150, data.y.shape[0])


class TestMoonsDataSet(unittest.TestCase):

    def test_init_moons_data_set(self):
        data = mls.datasets.Moons()
        self.assertIsNotNone(data)
        self.assertListEqual([], data.x)
        self.assertIsNotNone([], data.y)

    def test_generate_moons_data_set_is_generated_with_default_params(self):
        data = mls.datasets.Moons()
        data.generate()
        self.assertEqual(2, data.x.shape[1])
        self.assertEqual(100, data.x.shape[0])
        self.assertEqual(100, data.y.shape[0])

    def test_generate_moons_data_set_is_generated_by_factory(self):
        data = mls.datasets.DataSetFactory.create_dataset('Moons')
        data.generate()
        self.assertEqual(2, data.x.shape[1])
        self.assertEqual(100, data.x.shape[0])
        self.assertEqual(100, data.y.shape[0])


class TestCirclesDataSet(unittest.TestCase):

    def test_init_circles_data_set(self):
        data = mls.datasets.Circles()
        self.assertIsNotNone(data)
        self.assertListEqual([], data.x)
        self.assertIsNotNone([], data.y)

    def test_generate_moons_data_set_is_generated_with_default_params(self):
        data = mls.datasets.Circles()
        data.generate()
        self.assertEqual(2, data.x.shape[1])
        self.assertEqual(100, data.x.shape[0])
        self.assertEqual(100, data.y.shape[0])

    def test_generate_moons_data_set_is_generated_by_factory(self):
        data = mls.datasets.DataSetFactory.create_dataset('Circles')
        data.generate()
        self.assertEqual(2, data.x.shape[1])
        self.assertEqual(100, data.x.shape[0])
        self.assertEqual(100, data.y.shape[0])

import unittest

import mlsurvey as mls


class TestDataSet(unittest.TestCase):

    def test_set_generation_parameters(self):
        data = mls.datasets.DataSet('')
        params = {'param1': 1, 'param2': 3}
        data.set_generation_parameters(params)
        self.assertDictEqual(data.params, params)


class TestGenericDataEt(unittest.TestCase):

    def test_init_generic_data_set(self):
        data = mls.datasets.GenericDataSet('')
        self.assertIsNotNone(data)
        self.assertListEqual([], data.x)
        self.assertIsNotNone([], data.y)

    def test_set_generation_parameters(self):
        data = mls.datasets.GenericDataSet('load_iris')
        params = {'param1': 1, 'param2': 3, 'return_X_y': False}
        data.set_generation_parameters(params)
        self.assertDictEqual(data.params, params)

    def test_generate_generic_data_set_moon_is_generated(self):
        data = mls.datasets.DataSetFactory.create_dataset('make_moons')
        data.generate()
        self.assertEqual(2, data.x.shape[1])
        self.assertEqual(100, data.x.shape[0])
        self.assertEqual(100, data.y.shape[0])

    def test_generate_generic_data_set_iris_is_generated(self):
        data = mls.datasets.DataSetFactory.create_dataset('load_iris')
        data.generate()
        self.assertEqual(4, data.x.shape[1])
        self.assertEqual(5.1, data.x[0, 0])
        self.assertEqual(0, data.y[0])
        self.assertEqual(150, data.x.shape[0])
        self.assertEqual(150, data.y.shape[0])

    def test_generate_generic_data_set_classification_is_generated(self):
        data = mls.datasets.DataSetFactory.create_dataset('make_classification')
        data.generate()
        self.assertEqual(100, data.x.shape[0])
        self.assertEqual(20, data.x.shape[1])
        self.assertEqual(100, data.y.shape[0])

    def test_generate_generic_data_set_circle_is_generated_with_parameters(self):
        data = mls.datasets.DataSetFactory.create_dataset('make_circles')
        params = {
            'n_samples': 200,
            # not tested
            'shuffle': False,
            # not tested
            'noise': 0.5,
            # not tested
            'random_state': 10,
            # not tested
            'factor': 0.3
        }
        data.set_generation_parameters(params)
        data.generate()
        self.assertEqual(2, data.x.shape[1])
        self.assertEqual(params['n_samples'], data.x.shape[0])
        self.assertEqual(params['n_samples'], data.y.shape[0])

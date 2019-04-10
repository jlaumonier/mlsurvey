import unittest

import mlsurvey as mls


class TestDataSet(unittest.TestCase):

    def test_set_generation_parameters(self):
        data = mls.datasets.DataSet('')
        params = {'param1': 1, 'param2': 3}
        data.set_generation_parameters(params)
        self.assertDictEqual(data.params, params)


class TestGenericDataSet(unittest.TestCase):

    def test_init_generic_data_set(self):
        data = mls.datasets.GenericDataSet('')
        self.assertIsNotNone(data)

    def test_set_generation_parameters(self):
        data = mls.datasets.GenericDataSet('load_iris')
        params = {'param1': 1, 'param2': 3, 'return_X_y': False}
        data.set_generation_parameters(params)
        self.assertDictEqual(data.params, params)

    def test_generate_generic_dataset_moon_is_generated(self):
        data = mls.datasets.DataSetFactory.create_dataset('make_moons')
        (x, y) = data.generate()
        self.assertEqual(2, x.shape[1])
        self.assertEqual(100, x.shape[0])
        self.assertEqual(100, y.shape[0])

    def test_generate_generic_dataset_iris_is_generated(self):
        data = mls.datasets.DataSetFactory.create_dataset('load_iris')
        (x, y) = data.generate()
        self.assertEqual(4, x.shape[1])
        self.assertEqual(5.1, x[0, 0])
        self.assertEqual(0, y[0])
        self.assertEqual(150, x.shape[0])
        self.assertEqual(150, y.shape[0])

    def test_generate_generic_dataset_classification_is_generated(self):
        data = mls.datasets.DataSetFactory.create_dataset('make_classification')
        (x, y) = data.generate()
        self.assertEqual(100, x.shape[0])
        self.assertEqual(20, x.shape[1])
        self.assertEqual(100, y.shape[0])

    def test_generate_generic_dataset_circle_is_generated_with_parameters(self):
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
        (x, y) = data.generate()
        self.assertEqual(2, x.shape[1])
        self.assertEqual(params['n_samples'], x.shape[0])
        self.assertEqual(params['n_samples'], y.shape[0])

    def test_generate_generic_dataset_unknown_dataset_should_cause_error(self):
        """
        :test : mlsurvey.datasets.GenericDataset.generate()
        :condition : unknown dataset
        :main_result : raise AttributeError
        """
        data = mls.datasets.DataSetFactory.create_dataset('make_unknown')
        (x, y) = (None, None)
        try:
            (x, y) = data.generate()
            self.assertTrue(False)
        except AttributeError:
            self.assertTrue(True)
        self.assertIsNone(x)
        self.assertIsNone(y)

    def test_generate_generic_dataset_unknown_parameter_should_cause_error(self):
        """
        :test : mlsurvey.datasets.GenericDataset.generate()
        :condition : unknown parameter
        :main_result : raise TypeError
        """
        data = mls.datasets.DataSetFactory.create_dataset('make_circles')
        params = {
            'test': 200,
        }
        (x, y) = (None, None)
        data.set_generation_parameters(params)
        try:
            (x, y) = data.generate()
            self.assertTrue(False)
        except TypeError:
            self.assertTrue(True)
        self.assertIsNone(x)
        self.assertIsNone(y)

    def test_to_dict_transformed(self):
        dataset = mls.datasets.GenericDataSet('load_iris')
        params = {'param1': 1, 'param2': 3, 'return_X_y': False}
        expected = {'type': 'load_iris', "parameters": {'param1': 1, 'param2': 3, 'return_X_y': False}}
        dataset.set_generation_parameters(params)
        j = dataset.to_dict()
        self.assertDictEqual(j, expected)


class TestNClassRandomClassificationWithNoise(unittest.TestCase):

    def test_generate_nclassrandomclassificationwithnoise_unknown_dataset_should_generate(self):
        """
        :test : mlsurvey.datasets.NClassRandomClassificationWithNoise.generate()
        :condition : unknown parameter
        :main_result : Generate
        """
        data = mls.datasets.DataSetFactory.create_dataset('NClassRandomClassificationWithNoise')
        params = {
            'test': 200,
            'n_samples': 200
        }
        data.set_generation_parameters(params)
        (x, y) = data.generate()
        self.assertTrue(True)
        self.assertEqual(2, x.shape[1])
        self.assertEqual(params['n_samples'], x.shape[0])
        self.assertEqual(params['n_samples'], y.shape[0])

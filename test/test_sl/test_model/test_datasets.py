import unittest

import dask.dataframe as dd
import pandas as pd

import mlsurvey as mls


class TestDataSet(unittest.TestCase):

    def test_init_parameters_type_fairness(self):
        expected_type = 'typ'
        data = mls.sl.datasets.DataSet(expected_type)
        self.assertEqual(expected_type, data.t)
        self.assertEqual('Pandas', data.storage)
        self.assertDictEqual({}, data.params)
        self.assertDictEqual({}, data.fairness)
        self.assertDictEqual({'y_col_name': 'target'}, data.metadata)

    def test_init_parameters_type_fairness_storage(self):
        expected_type = 'typ'
        data = mls.sl.datasets.DataSet(expected_type, 'Dask')
        self.assertEqual(expected_type, data.t)
        self.assertEqual('Dask', data.storage)
        self.assertDictEqual({}, data.params)
        self.assertDictEqual({}, data.fairness)
        self.assertDictEqual({'y_col_name': 'target'}, data.metadata)

    def test_set_generation_parameters(self):
        data = mls.sl.datasets.DataSet('')
        params = {'param1': 1, 'param2': 3}
        data.set_generation_parameters(params)
        self.assertDictEqual(data.params, params)

    def test_set_fairness_parameters(self):
        data = mls.sl.datasets.DataSet('')
        expected_fairness = {'protected_attribute': 0, 'privileged_classes': 'x >= 25'}
        data.set_fairness_parameters(expected_fairness)
        self.assertDictEqual(data.fairness, expected_fairness)

    def test_set_metadata_parameters_set(self):
        dataset = mls.sl.datasets.DataSet('')
        metadata = {'y_col_name': 'col1'}
        dataset.set_metadata_parameters(metadata)
        self.assertDictEqual(dataset.metadata, metadata)

    def test_set_metadata_parameters_empty_should_have_default_y_col_name(self):
        dataset = mls.sl.datasets.DataSet('')
        metadata = None
        dataset.set_metadata_parameters(metadata)
        self.assertEqual('target', dataset.metadata['y_col_name'])

    def test_set_metadata_parameters_no_y_col_name_should_have_default_y_col_name(self):
        dataset = mls.sl.datasets.DataSet('')
        metadata = {}
        dataset.set_metadata_parameters(metadata)
        self.assertEqual('target', dataset.metadata['y_col_name'])

    def test_to_dict_is_transformed(self):
        data = mls.sl.datasets.DataSet('')
        params = {'param1': 1, 'param2': 3}
        data.set_generation_parameters(params)
        fairness = {'protected_attribute': 0, 'privileged_classes': 'x >= 25'}
        data.set_fairness_parameters(fairness)
        metadata = {'y_col_name': 'col1'}
        data.set_metadata_parameters(metadata)
        dict_result = data.to_dict()
        expected = {'type': '', 'storage': 'Pandas', 'parameters': params, 'fairness': fairness, 'metadata': metadata}
        self.assertDictEqual(expected, dict_result)

    def test_set_fairness_parameters_empty_parameter(self):
        """
        :test : mlsurvey.datasets.DataSet.set_fairness_parameters
        :condition : empty fairness parameters
        :main_result : raise ConfigError
        """
        data = mls.sl.datasets.DataSet('')
        expected_fairness = {}
        try:
            data.set_fairness_parameters(expected_fairness)
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_set_fairness_parameters_none_parameter(self):
        """
        :test : mlsurvey.datasets.DataSet.set_fairness_parameters
        :condition : none fairness parameters
        :main_result : raise ConfigError
        """
        data = mls.sl.datasets.DataSet('')
        expected_fairness = None
        try:
            data.set_fairness_parameters(expected_fairness)
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_set_fairness_parameters_no_protected_parameter(self):
        """
        :test : mlsurvey.datasets.DataSet.set_fairness_parameters
        :condition : fairness parameters does not contains a protected_attribute
        :main_result : raise ConfigError
        """
        data = mls.sl.datasets.DataSet('')
        expected_fairness = {'param1': 1, 'param2': 3}
        try:
            data.set_fairness_parameters(expected_fairness)
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_set_fairness_parameters_no_privileged_classes_parameter(self):
        """
        :test : mlsurvey.datasets.DataSet.set_fairness_parameters
        :condition : fairness parameters does not contains privileged_classes
        :main_result : raise ConfigError
        """
        data = mls.sl.datasets.DataSet('')
        expected_fairness = {'param1': 1, 'param2': 3, 'protected_attribute': 0}
        try:
            data.set_fairness_parameters(expected_fairness)
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)


class TestGenericDataSet(unittest.TestCase):

    def test_init_generic_data_set(self):
        data = mls.sl.datasets.GenericDataSet('')
        self.assertIsNotNone(data)

    def test_set_generation_parameters(self):
        data = mls.sl.datasets.GenericDataSet('load_iris')
        params = {'param1': 1, 'param2': 3, 'return_X_y': False}
        data.set_generation_parameters(params)
        self.assertDictEqual(params, data.params)

    def test_generate_generic_dataset_moon_is_generated(self):
        moon_dataset = mls.sl.datasets.DataSetFactory.create_dataset('make_moons')
        moon_data = moon_dataset.generate()
        self.assertIsInstance(moon_data, pd.DataFrame)
        self.assertEqual(2, moon_data.iloc[:, 0:-1].shape[1])
        self.assertEqual(100, moon_data.iloc[:, 0:-1].shape[0])
        self.assertEqual(100, moon_data.iloc[:, -1].shape[0])

    def test_generate_generic_dataset_iris_is_generated(self):
        iris_dataset = mls.sl.datasets.DataSetFactory.create_dataset('load_iris')
        iris_data = iris_dataset.generate()
        self.assertEqual(4, iris_data.iloc[:, 0:-1].shape[1])
        self.assertEqual(5.1, iris_data.iloc[0, 0])
        self.assertEqual(0, iris_data.iloc[0, -1])
        self.assertEqual(150, iris_data.iloc[:, 0:-1].shape[0])
        self.assertEqual(150, iris_data.iloc[:, -1].shape[0])

    def test_generate_generic_dataset_classification_is_generated(self):
        classification_dataset = mls.sl.datasets.DataSetFactory.create_dataset('make_classification')
        classification_data = classification_dataset.generate()
        self.assertEqual(100, classification_data.iloc[:, 0:-1].shape[0])
        self.assertEqual(20, classification_data.iloc[:, 0:-1].shape[1])
        self.assertEqual(100, classification_data.iloc[:, -1].shape[0])

    def test_generate_generic_dataset_circle_is_generated_with_parameters(self):
        circle_dataset = mls.sl.datasets.DataSetFactory.create_dataset('make_circles')
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
        circle_dataset.set_generation_parameters(params)
        circle_data = circle_dataset.generate()
        self.assertIsInstance(circle_data, pd.DataFrame)
        self.assertEqual(2, circle_data.iloc[:, 0:-1].shape[1])
        self.assertEqual(params['n_samples'], circle_data.iloc[:, 0:-1].shape[0])
        self.assertEqual(params['n_samples'], circle_data.iloc[:, -1].shape[0])

    def test_generate_generic_dataset_circle_is_generated_with_parameters_dask(self):
        circle_dataset = mls.sl.datasets.DataSetFactory.create_dataset('make_circles', 'Dask')
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
        circle_dataset.set_generation_parameters(params)
        circle_data = circle_dataset.generate()
        self.assertIsInstance(circle_data, dd.DataFrame)
        self.assertEqual(2, circle_data.iloc[:, 0:-1].shape[1])
        self.assertEqual(params['n_samples'], circle_data.iloc[:, 0:-1].compute().shape[0])
        self.assertEqual(params['n_samples'], circle_data.iloc[:, -1].shape[0])

    def test_generate_generic_dataset_unknown_dataset_should_cause_error(self):
        """
        :test : mlsurvey.datasets.GenericDataset.generate()
        :condition : unknown dataset
        :main_result : raise AttributeError
        """
        data = mls.sl.datasets.DataSetFactory.create_dataset('make_unknown')
        df = None
        try:
            df = data.generate()
            self.assertTrue(False)
        except AttributeError:
            self.assertTrue(True)
        self.assertIsNone(df)

    def test_generate_generic_dataset_unknown_parameter_should_cause_error(self):
        """
        :test : mlsurvey.datasets.GenericDataset.generate()
        :condition : unknown parameter
        :main_result : raise TypeError
        """
        data = mls.sl.datasets.DataSetFactory.create_dataset('make_circles')
        params = {
            'test': 200,
        }
        df = None
        data.set_generation_parameters(params)
        try:
            df = data.generate()
            self.assertTrue(False)
        except TypeError:
            self.assertTrue(True)
        self.assertIsNone(df)

    def test_to_dict_transformed(self):
        dataset = mls.sl.datasets.GenericDataSet('load_iris')
        params = {'param1': 1, 'param2': 3, 'return_X_y': False}
        expected = {'type': 'load_iris',
                    'storage': 'Pandas',
                    'parameters': {'param1': 1, 'param2': 3, 'return_X_y': False},
                    'metadata': {'y_col_name': 'target'}}
        dataset.set_generation_parameters(params)
        j = dataset.to_dict()
        self.assertDictEqual(expected, j)


class TestNClassRandomClassificationWithNoise(unittest.TestCase):

    def test_generate_nclassrandomclassificationwithnoise_unknown_dataset_should_generate(self):
        """
        :test : mlsurvey.datasets.NClassRandomClassificationWithNoise.generate()
        :condition : unknown parameter
        :main_result : Generate
        """
        data = mls.sl.datasets.DataSetFactory.create_dataset('NClassRandomClassificationWithNoise')
        params = {
            'test': 200,
            'n_samples': 200
        }
        data.set_generation_parameters(params)
        df = data.generate()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(2, df.iloc[:, 0:-1].shape[1])
        self.assertEqual(params['n_samples'], df.shape[0])

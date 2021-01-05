import os
import unittest

import pandas as pd

import mlsurvey as mls


class TestFileDataSet(unittest.TestCase):
    d = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.d = os.path.join(directory, '../../files/dataset/')
        cls.base_dir = os.path.join(directory, '../../')

    def test_init_should_init(self):
        """
        :test : mlsurvey.Datasets.FileDataset.load_file()
        :condition : filename is given
        :main_result : dataset is init
        """
        dataset = mls.sl.datasets.DataSetFactory.create_dataset('FileDataSet')
        self.assertIsInstance(dataset, mls.sl.datasets.FileDataSet)
        self.assertEqual('', dataset.base_directory)

    def test_generate_file_dataset_german_credit_generated(self):
        filename = 'credit-g.arff'
        dataset = mls.sl.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': self.d, 'filename': filename}
        expected_columns = ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
                            'savings_status', 'employment', 'installment_commitment', 'personal_status',
                            'other_parties', 'residence_since', 'property_magnitude', 'age', 'other_payment_plans',
                            'housing', 'existing_credits', 'job', 'num_dependents', 'own_telephone', 'foreign_worker',
                            'class']
        dataset.set_generation_parameters(params)
        df = dataset.generate()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(1000, df.iloc[:, 0:-1].shape[0])
        self.assertEqual(20, df.iloc[:, 0:-1].shape[1])
        self.assertEqual(1000, df.iloc[:, -1].shape[0])
        self.assertEqual(1, df.iloc[:, -1].ndim)
        self.assertListEqual(expected_columns, df.keys().to_list())

    def test_generate_file_dataset_german_credit_generated_with_relative_path(self):
        filename = 'credit-g.arff'
        dataset = mls.sl.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': 'files/dataset/', 'filename': filename}
        dataset.set_generation_parameters(params)
        dataset.set_base_directory(self.base_dir)
        df = dataset.generate()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(1000, df.iloc[:, 0:-1].shape[0])
        self.assertEqual(20, df.iloc[:, 0:-1].shape[1])
        self.assertEqual(1000, df.iloc[:, -1].shape[0])
        self.assertEqual(1, df.iloc[:, -1].ndim)

    def test_generate_file_dataset_directory_not_present(self):
        filename = 'credit-g.arff'
        dataset = mls.sl.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'d': self.d, 'filename': filename}
        dataset.set_generation_parameters(params)
        try:
            _ = dataset.generate()
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_generate_file_dataset_filename_not_present(self):
        filename = 'credit-g.arff'
        dataset = mls.sl.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': self.d, 'f': filename}
        dataset.set_generation_parameters(params)
        try:
            _ = dataset.generate()
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_generate_file_dataset_file_not_found(self):
        filename = 'credit.arff'
        dataset = mls.sl.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': self.d, 'filename': filename}
        dataset.set_generation_parameters(params)
        try:
            _ = dataset.generate()
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_generate_file_dataset_unittest_de_csv(self):
        filename = 'unittest_de.csv'
        dataset = mls.sl.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': self.d, 'filename': filename}
        expected_columns = ['userid', 'offerid', 'countrycode', 'category', 'merchant', 'utcdate', 'rating']
        dataset.set_generation_parameters(params)
        df = dataset.generate()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(39, df.iloc[:, 0:-1].shape[0])
        self.assertEqual(6, df.iloc[:, 0:-1].shape[1])
        self.assertEqual(39, df.iloc[:, -1].shape[0])
        self.assertEqual(1, df.iloc[:, -1].ndim)
        self.assertListEqual(expected_columns, df.keys().to_list())

    def test_generate_file_dataset_xlsx_no_parameter(self):
        """
        :test : mlsurvey.sl.datasets.generate()
        :condition : load xslx file with no loading parameter
        :main_result : file loaded
        """
        filename = 'test.xlsx'
        dataset = mls.sl.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': self.d, 'filename': filename}
        expected_columns = ['Sheet', 'Column1', 'Column2 ', 'Column3', 'Column 4']
        dataset.set_generation_parameters(params)
        df = dataset.generate()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(expected_columns, df.keys().to_list())
        self.assertEqual(df['Column1'][0], 'A')
        self.assertEqual(df['Column2 '][1], 'Longer sentence.')

    def test_generate_file_dataset_xlsx_with_parameter(self):
        """
        :test : mlsurvey.sl.datasets.generate()
        :condition : load xslx file with loading parameters
        :main_result : file loaded. sheet merged, one column is added with sheet's name
        """
        filename = 'test.xlsx'
        dataset = mls.sl.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': self.d, 'filename': filename}
        expected_columns = ['Sheet', 'Column1', 'Column2 ', 'Column3', 'Column 4']
        dataset.set_generation_parameters(params)
        df = dataset.generate()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(expected_columns, df.keys().to_list())
        self.assertEqual('A', df['Column1'][0])
        self.assertEqual('2ndSheet2', df['Column2 '][2])
        self.assertEqual('RRRR', df['Column 4'][2])
        self.assertEqual('OtherSheet ', df['Sheet'][2])
        self.assertEqual('1stSheet', df['Sheet'][0])

    def test_generate_file_dataset_json(self):
        """
        :test : mlsurvey.sl.datasets.generate()
        :condition : load json file
        :main_result : file loaded
        """
        filename = '../../files/dataset/test-data.json'
        dataset = mls.sl.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': self.d, 'filename': filename, 'func_params': {'encoding': 'utf8', 'orient': 'index'}}
        expected_columns = ['Column1', 'Column2 ', 'Column3', 'Column 4']
        dataset.set_generation_parameters(params)
        df = dataset.generate()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(expected_columns, df.keys().to_list())
        self.assertEqual(df['Column1'][0], 'A')
        self.assertEqual(df['Column2 '][1], 'Longer sentence.')

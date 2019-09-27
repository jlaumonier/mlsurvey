import os
import unittest

import pandas as pd

import mlsurvey as mls


class TestFileDataSet(unittest.TestCase):
    d = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.d = os.path.join(directory, '../files/dataset/')
        cls.base_dir = os.path.join(directory, '../')

    def test_init_should_init(self):
        """
        :test : mlsurvey.Datasets.FileDataset.load_file()
        :condition : filename is given
        :main_result : dataset is init
        """
        dataset = mls.datasets.DataSetFactory.create_dataset('FileDataSet')
        self.assertIsInstance(dataset, mls.datasets.FileDataSet)
        self.assertEqual('', dataset.base_directory)

    def test_generate_file_dataset_german_credit_generated(self):
        filename = 'credit-g.arff'
        dataset = mls.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': self.d, 'filename': filename}
        dataset.set_generation_parameters(params)
        df = dataset.generate()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(1000, df.iloc[:, 0:-1].shape[0])
        self.assertEqual(20, df.iloc[:, 0:-1].shape[1])
        self.assertEqual(1000, df.iloc[:, -1].shape[0])
        self.assertEqual(1, df.iloc[:, -1].ndim)

    def test_generate_file_dataset_german_credit_generated_with_relative_path(self):
        filename = 'credit-g.arff'
        dataset = mls.datasets.DataSetFactory.create_dataset('FileDataSet')
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
        dataset = mls.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'d': self.d, 'filename': filename}
        dataset.set_generation_parameters(params)
        try:
            _ = dataset.generate()
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_generate_file_dataset_filename_not_present(self):
        filename = 'credit-g.arff'
        dataset = mls.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': self.d, 'f': filename}
        dataset.set_generation_parameters(params)
        try:
            _ = dataset.generate()
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_generate_file_dataset_file_not_found(self):
        filename = 'credit.arff'
        dataset = mls.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': self.d, 'filename': filename}
        dataset.set_generation_parameters(params)
        try:
            _ = dataset.generate()
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_generate_file_dataset_unittest_de_csv(self):
        filename = 'unittest_de.csv'
        dataset = mls.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': self.d, 'filename': filename}
        dataset.set_generation_parameters(params)
        df = dataset.generate()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(39, df.iloc[:, 0:-1].shape[0])
        self.assertEqual(6, df.iloc[:, 0:-1].shape[1])
        self.assertEqual(39, df.iloc[:, -1].shape[0])
        self.assertEqual(1, df.iloc[:, -1].ndim)

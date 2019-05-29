import os
import unittest

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
        (x, y) = dataset.generate()
        self.assertEqual(1000, x.shape[0])
        self.assertEqual(20, x.shape[1])
        self.assertEqual(1000, y.shape[0])
        self.assertEqual(1, y.ndim)

    def test_generate_file_dataset_german_credit_generated_with_relative_path(self):
        filename = 'credit-g.arff'
        dataset = mls.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': 'files/dataset/', 'filename': filename}
        dataset.set_generation_parameters(params)
        dataset.set_base_directory(self.base_dir)
        (x, y) = dataset.generate()
        self.assertEqual(1000, x.shape[0])
        self.assertEqual(20, x.shape[1])
        self.assertEqual(1000, y.shape[0])
        self.assertEqual(1, y.ndim)

    def test_generate_file_dataset_directory_not_present(self):
        filename = 'credit-g.arff'
        dataset = mls.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'d': self.d, 'filename': filename}
        dataset.set_generation_parameters(params)
        try:
            _, _ = dataset.generate()
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_generate_file_dataset_filename_not_present(self):
        filename = 'credit-g.arff'
        dataset = mls.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': self.d, 'f': filename}
        dataset.set_generation_parameters(params)
        try:
            _, _ = dataset.generate()
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

    def test_generate_file_dataset_file_not_found(self):
        filename = 'credit.arff'
        dataset = mls.datasets.DataSetFactory.create_dataset('FileDataSet')
        params = {'directory': self.d, 'filename': filename}
        dataset.set_generation_parameters(params)
        try:
            _, _ = dataset.generate()
            self.assertTrue(False)
        except mls.exceptions.ConfigError:
            self.assertTrue(True)

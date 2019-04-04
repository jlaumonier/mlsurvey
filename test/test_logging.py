import os
import shutil
import unittest

from sklearn import neighbors

import mlsurvey as mls


class TestLogging(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir)

    def test_init_log_directory_created_with_date(self):
        log = mls.Logging()
        self.assertIsNotNone(log.dir_name)
        self.assertTrue(os.path.isdir(log.base_dir + log.dir_name + '/'))
        self.assertEqual(log.directory, log.base_dir + log.dir_name + '/')

    def test_init_log_directory_create_with_fixed_name(self):
        dir_name = 'testing/'
        log = mls.Logging(dir_name=dir_name)
        self.assertEqual(log.directory, 'logs/' + dir_name)
        self.assertTrue(os.path.isdir('logs/' + dir_name))

    def test_save_inputs_input_saved(self):
        dir_name = 'testing/'
        d_data = mls.datasets.DataSetFactory.create_dataset('make_moons')
        d_data.params['random_state'] = 0
        i_data = mls.models.Data()
        i_data.x, i_data.y = d_data.generate()
        d_train = mls.datasets.DataSetFactory.create_dataset('load_iris')
        i_train = mls.models.Data()
        i_train.x, i_train.y = d_train.generate()
        d_test = mls.datasets.DataSetFactory.create_dataset('make_circles')
        d_test.params['random_state'] = 0
        i_test = mls.models.Data()
        i_test.x, i_test.y = d_test.generate()
        log = mls.Logging(dir_name)
        inputs = {'data': i_data, 'train': i_train, 'test': i_test}
        log.save_input(inputs)
        self.assertTrue(os.path.isfile(log.directory + 'input.json'))
        self.assertEqual('8c49ab773ad380d4680471b0c6df8e92', mls.Utils.md5_file(log.directory + 'input.json'))

    def test_load_input_input_loaded(self):
        dir_name = 'files/'
        log = mls.Logging(dir_name, base_dir='../test/')
        results = log.load_input('logging_test_load_input_input_loaded.json')
        i = results['test']
        self.assertEqual(5.1, i.x[0, 0])
        self.assertEqual(0, i.y[0])
        self.assertEqual(2, i.x.shape[1])
        self.assertEqual(150, i.x.shape[0])
        self.assertEqual(150, i.y.shape[0])

    def test_save_json_file_saves(self):
        dir_name = 'testing/'
        log = mls.Logging(dir_name)
        d = {'testA': [[1, 2], [3, 4]], 'testB': 'Text'}
        log.save_dict_as_json('dict.json', d)
        self.assertTrue(os.path.isfile(log.directory + 'dict.json'))
        self.assertEqual('a82076220e033c1ed3469d173d715df2', mls.Utils.md5_file(log.directory + 'dict.json'))

    def test_load_json_dict_loaded(self):
        dir_name = 'files/'
        log = mls.Logging(dir_name, base_dir='../test/')
        result = log.load_json_as_dict('dict.json')
        expected = {'testA': [[1, 2], [3, 4]], 'testB': 'Text'}
        self.assertDictEqual(expected, result)

    def test_save_classifier(self):
        dir_name = 'testing/'
        log = mls.Logging(dir_name)
        classifier = neighbors.KNeighborsClassifier()
        log.save_classifier(classifier)
        self.assertTrue(os.path.isfile(log.directory + 'model.joblib'))
        self.assertEqual('6c9bde80d5b0e3c3adb0b6822c692e14', mls.Utils.md5_file(log.directory + 'model.joblib'))

    def test_load_classifier(self):
        dir_name = 'files/slw'
        log = mls.Logging(dir_name, base_dir='../test/')
        classifier = log.load_classifier()
        self.assertIsInstance(classifier, neighbors.KNeighborsClassifier)

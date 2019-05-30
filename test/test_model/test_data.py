import unittest

import numpy as np

import mlsurvey as mls


class TestData(unittest.TestCase):

    def test_init_data(self):
        i = mls.models.Data()
        np.testing.assert_array_equal(np.array([]), i.x)
        np.testing.assert_array_equal(np.array([]), i.y)
        np.testing.assert_array_equal(np.array([]), i.y_pred)

    def test_to_dict_dict_should_be_set(self):
        d = mls.models.Data()
        d.x = np.array([[1, 2], [3, 4]])
        d.y = np.array([0, 1])
        d.y_pred = np.array([1, 0])
        expected = {'data.x': [[1, 2], [3, 4]], 'data.y': [0, 1], 'data.y_pred': [1, 0]}
        result = d.to_dict()
        self.assertDictEqual(expected, result)

    def test_from_dict_input_is_set(self):
        input_dict = {'data.x': [[1, 2], [3, 4]], 'data.y': [0, 1], 'data.y_pred': [1, 0]}
        d = mls.models.Data.from_dict(input_dict)
        self.assertEqual(1, d.x[0, 0])
        self.assertEqual(0, d.y[0])
        self.assertEqual(1, d.y_pred[0])
        self.assertEqual(2, d.x.shape[1])
        self.assertEqual(2, d.x.shape[0])
        self.assertEqual(2, d.y.shape[0])
        self.assertEqual(2, d.y_pred.shape[0])

    def test_from_dict_datax_not_present(self):
        """ maybe this should be test using a json schema validator """
        input_dict = {'d.x': [[1, 2], [3, 4]], 'data.y': [0, 1], 'data.y_pred': [1, 0]}
        d = None
        try:
            d = mls.models.Data.from_dict(input_dict)
            self.assertTrue(False)
        except mls.exceptions.ModelError:
            self.assertIsNone(d)
            self.assertTrue(True)

    def test_from_dict_datay_not_present(self):
        """ maybe this should be test using a json schema validator """
        input_dict = {'data.x': [[1, 2], [3, 4]], 'd.y': [0, 1], 'data.y_pred': [1, 0]}
        d = None
        try:
            d = mls.models.Data.from_dict(input_dict)
            self.assertTrue(False)
        except mls.exceptions.ModelError:
            self.assertIsNone(d)
            self.assertTrue(True)

    def test_from_dict_dataypred_not_present(self):
        """ maybe this should be test using a json schema validator """
        input_dict = {'data.x': [[1, 2], [3, 4]], 'data.y': [0, 1], 'd.y_pred': [1, 0]}
        d = None
        try:
            d = mls.models.Data.from_dict(input_dict)
            self.assertTrue(False)
        except mls.exceptions.ModelError:
            self.assertIsNone(d)
            self.assertTrue(True)

    def test_merge_all_should_merge(self):
        """
        :test : mlsurvey.modles.Data.merge_all()
        :condition : data contains x, y and y_pred
        :main_result : data are merge into one array
        """
        d = mls.models.Data()
        d.x = np.array([[1, 2], [3, 4]])
        d.y = np.array([0, 1])
        d.y_pred = np.array([1, 0])
        expected_result = np.asarray([[1, 2, 0, 1],
                                      [3, 4, 1, 0]])
        result = d.merge_all()
        np.testing.assert_array_equal(expected_result, result)

    def test_copy_data_should_copy(self):
        """
        :test : mlsurvey.modles.Data.copy()
        :condition : data contains x, y and y_pred
        :main_result : copy into an other object
        """
        d = mls.models.Data()
        d.x = np.array([[1, 2], [3, 4]])
        d.y = np.array([0, 1])
        d.y_pred = np.array([1, 0])
        d_copied = d.copy()
        np.testing.assert_array_equal(d.x, d_copied.x)
        np.testing.assert_array_equal(d.y, d_copied.y)
        np.testing.assert_array_equal(d.y_pred, d_copied.y_pred)
        self.assertIsNot(d, d_copied)

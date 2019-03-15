import unittest

import sklearn.neighbors as neighbors

import mlsurvey as mls


class TestUtils(unittest.TestCase):

    def test_md5_file(self):
        md5 = mls.Utils.md5_file('files/test_md5.txt')
        self.assertEqual('70a4b9f4707d258f559f91615297a3ec', md5)

    def test_dict_generator_cartesian_product(self):
        source = {"element1": ["A", "B", "C"], "element2": [1, 2], "element3": [True, False], "element4": 12}
        result = list(mls.Utils.dict_generator_cartesian_product(source))
        self.assertEqual(12, len(result))
        expected_first_dict = {"element1": "A", "element2": 1, "element3": True, "element4": 12}
        self.assertDictEqual(expected_first_dict, result[0])
        expected_second_dict = {"element1": "A", "element2": 1, "element3": False, "element4": 12}
        self.assertDictEqual(expected_second_dict, result[1])

    def test_dict_generator_cartesian_product_string_not_list(self):
        source = {"element1": "test"}
        result = list(mls.Utils.dict_generator_cartesian_product(source))
        self.assertEqual(1, len(result))
        expected_first_dict = {"element1": "test"}
        self.assertDictEqual(expected_first_dict, result[0])

    def test_dict_generator_cartesian_product_empty(self):
        source = {}
        result = list(mls.Utils.dict_generator_cartesian_product(source))
        self.assertEqual(1, len(result))
        expected_first_dict = {}
        self.assertDictEqual(expected_first_dict, result[0])

    def test_import_from_dotted_path_class_created(self):
        to_import = 'sklearn.neighbors.KNeighborsClassifier'
        classdef = mls.Utils.import_from_dotted_path(to_import)
        self.assertEqual(neighbors.KNeighborsClassifier, classdef)

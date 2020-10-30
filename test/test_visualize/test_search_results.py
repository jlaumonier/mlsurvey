import os
import unittest

import tinydb as tdb

import mlsurvey as mls


class TestSearchResults(unittest.TestCase):
    directory = ''
    analyse_logs = None

    @classmethod
    def setUpClass(cls):
        d = os.path.dirname(__file__)
        cls.directory = os.path.join(d, '../files/visualize-log//')
        cls.analyse_logs = mls.visualize.AnalyzeLogs(cls.directory)
        cls.analyse_logs.store_config()
        cls.analyse_logs.fill_lists()

    def test_select_criteria_value_operator_algorithm_family(self):
        """
        :test : mlsurvey.visualize.SearchInterface.select_criteria_value_operator
        :condition : config files present in self.directory, criteria is 'algorithm.type'
        :main_result : all possible values and operator are returned
        """
        search_interface = mls.visualize.SearchInterface(self.analyse_logs)
        criteria = 'algorithm.type'
        result_value, result_operator = search_interface.select_criteria_value_operator(criteria)
        expected_result_value = [{'label': 'sklearn.neighbors.KNeighborsClassifier',
                                  'value': 'sklearn.neighbors.KNeighborsClassifier'},
                                 {'label': 'sklearn.svm.SVC',
                                  'value': 'sklearn.svm.SVC'},
                                 {'label': 'sklearn.neural_network.MLPClassifier',
                                  'value': 'sklearn.neural_network.MLPClassifier'}
                                 ]
        expected_result_operator = [{'label': '==',
                                     'value': '=='},
                                    {'label': '!=',
                                     'value': '!='}
                                    ]
        self.assertListEqual(expected_result_value, result_value)
        self.assertListEqual(expected_result_operator, result_operator)

    def test_select_criteria_value_operator_n_neighbors(self):
        """
        :test : mlsurvey.visualize.SearchInterface.select_criteria_value_operator
        :condition : config files present in self.directory, criteria is 'algorithm.hyperparameters.n_neighbors'
        :main_result : all possible values and operator are returned
        """
        search_interface = mls.visualize.SearchInterface(self.analyse_logs)
        criteria = 'algorithm.hyperparameters.n_neighbors'
        result_value, result_operator = search_interface.select_criteria_value_operator(criteria)
        expected_result_value = [{'label': '2.0',
                                  'value': '2.0'},
                                 {'label': 'nan',
                                  'value': 'nan'}
                                 ]
        expected_result_operator = [{'label': '==',
                                     'value': '=='},
                                    {'label': '!=',
                                     'value': '!='},
                                    {'label': '<=',
                                     'value': '<='},
                                    {'label': '>=',
                                     'value': '>='},
                                    {'label': '<',
                                     'value': '<'},
                                    {'label': '>',
                                     'value': '>'}
                                    ]
        self.assertListEqual(expected_result_value, result_value)
        self.assertListEqual(expected_result_operator, result_operator)

    def test_select_criteria_value_operator_no_criteria(self):
        """
        :test : mlsurvey.visualize.SearchInterface.select_criteria_value_operator
        :condition : config files present in self.directory, criteria is None or ''
        :main_result : value and operator are None
        """
        search_interface = mls.visualize.SearchInterface(self.analyse_logs)
        criteria = ''
        result_value, result_operator = search_interface.select_criteria_value_operator(criteria)
        self.assertIsNone(result_value)
        self.assertIsNone(result_operator)
        criteria = None
        result_value, result_operator = search_interface.select_criteria_value_operator(criteria)
        self.assertIsNone(result_value)
        self.assertIsNone(result_operator)

    def test_add_criteria(self):
        """
        :test : mlsurvey.visualize.SearchInterface.add_criteria
        :condition : criteria is algorithm.type==toto
                    and expected_existing_options
                    and expected_existing_values are empty
        :main_result : existing_options are correctly existing_values
        """
        search_interface = mls.visualize.SearchInterface(self.analyse_logs)
        criteria_operator = '=='
        criteria_value = 'toto'
        criteria = 'algorithm.type'
        existing_options = []
        existing_values = []
        expected_existing_options = [{'label': 'algorithm.type==toto',
                                      'value': "{'criteria': 'algorithm.type', "
                                               "'criteria_operator': '==', "
                                               "'criteria_value': 'toto'}"}]
        expected_existing_values = ["{'criteria': 'algorithm.type', "
                                    "'criteria_operator': '==', "
                                    "'criteria_value': 'toto'}"]
        existing_options, existing_values = search_interface.add_criteria(0,
                                                                          criteria_value,
                                                                          criteria_operator,
                                                                          criteria,
                                                                          existing_options,
                                                                          existing_values)
        self.assertListEqual(expected_existing_options, existing_options)
        self.assertListEqual(expected_existing_values, existing_values)

    def test_search_one_result(self):
        """
        :test : mlsurvey.visualize.SearchInterface.search()
        :condition : criteria_values is algorithm.type==sklearn.neighbors.KNeighborsClassifier
        :main_result : one result in list defined as list of dictionary
        """
        search_interface = mls.visualize.SearchInterface(self.analyse_logs)
        value_algo = '.'
        value_ds = '.'
        criteria_values = ["{'criteria': 'algorithm.type', "
                           "'criteria_operator': '==', "
                           "'criteria_value': 'sklearn.neighbors.KNeighborsClassifier'}"]
        result, search_result = search_interface.search(value_algo, value_ds, criteria_values)
        expected_result = [{'input': 'NClassRandomClassificationWithNoise',
                            'inputParams': "{'n_samples': 100, 'shuffle': True, 'noise': 0, 'random_state': 0}",
                            'split': "traintest",
                            "splitParams": "{'test_size': 20, 'random_state': 0, 'shuffle': True}",
                            'algorithm': 'sklearn.neighbors.KNeighborsClassifier',
                            'algorithmParams': "{'n_neighbors': 2, 'algorithm': 'auto', 'weights': 'uniform'}",
                            'FairnessParams': '{}',
                            'Directory': self.directory + 'directory1',
                            'Type': 'mlsurvey.sl.workflows.SupervisedLearningWorkflow'}]
        self.assertListEqual(expected_result, result)
        self.assertEqual(1, len(search_result))

    def test_search_two_results(self):
        """
        :test : mlsurvey.visualize.SearchInterface.search()
        :condition : criteria_values is algorithm.type==sklearn.svm.SVC
        :main_result : one result in list defined as list of dictionary
        """
        search_interface = mls.visualize.SearchInterface(self.analyse_logs)
        value_algo = '.'
        value_ds = '.'
        criteria_values = ["{'criteria': 'algorithm.type', "
                           "'criteria_operator': '==', "
                           "'criteria_value': 'sklearn.svm.SVC'}"]
        result, search_result = search_interface.search(value_algo, value_ds, criteria_values)
        self.assertEqual(2, len(result))
        self.assertEqual(2, len(search_result))

    def test_get_result_figure_summary_two_results(self):
        """
        :test : mlsurvey.visualize.SearchInterface.list_result_figure()
        :condition : criteria_values is algorithm.type==sklearn.svm.SVC
        :main_result : result for C (x) and score (y)
        """
        search_interface = mls.visualize.SearchInterface(self.analyse_logs)
        query = tdb.Query()
        search_result = self.analyse_logs.db.search(
            query.learning_process.parameters.algorithm['type'].matches('sklearn.svm.SVC'))
        result_df, list_of_not_unique_key = search_interface.get_result_figure_summary(search_result)
        self.assertListEqual([0.1, 0.7], result_df['algorithm.hyperparameters.C'].to_list())
        self.assertListEqual([1.0, 1.0], result_df['score'].to_list())
        self.assertListEqual(['algorithm.hyperparameters.C'], list_of_not_unique_key)

    def test_get_result_figure_summary_zero_results_too_many_not_unique(self):
        """
        :test : mlsurvey.visualize.SearchInterface.list_result_figure()
        :condition : no criteria_values
        :main_result : no result for x and y
        """
        search_interface = mls.visualize.SearchInterface(self.analyse_logs)
        search_result = self.analyse_logs.db.all()
        result_df, list_of_not_unique_key = search_interface.get_result_figure_summary(search_result)
        expected_list_not_unique_key = ['input.parameters.n_samples',
                                        'algorithm.type',
                                        'algorithm.hyperparameters.n_neighbors',
                                        'algorithm.hyperparameters.algorithm',
                                        'algorithm.hyperparameters.weights',
                                        'algorithm.hyperparameters.C',
                                        'algorithm.hyperparameters.kernel',
                                        'algorithm.hyperparameters.gamma',
                                        'algorithm.hyperparameters.activation',
                                        'algorithm.hyperparameters.max_iter'
                                        ]
        self.assertListEqual([], result_df['x'].to_list())
        self.assertListEqual([], result_df['y'].to_list())
        self.assertListEqual(expected_list_not_unique_key, list_of_not_unique_key)

    def test_get_result_figure_summary_zero_result(self):
        """
        :test : mlsurvey.visualize.SearchInterface.list_result_figure()
        :condition : criteria_values is algorithm.type==NoAlgo
        :main_result : empyt result
        """
        search_interface = mls.visualize.SearchInterface(self.analyse_logs)
        query = tdb.Query()
        search_result = self.analyse_logs.db.search(
            query.learning_process.algorithm['type'].matches('NoAlgo'))
        result_df, _ = search_interface.get_result_figure_summary(search_result)
        self.assertListEqual([], result_df['x'].to_list())
        self.assertListEqual([], result_df['y'].to_list())

import os
import unittest

import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
from sklearn import neighbors

import mlsurvey as mls


class TestVisualizeLogSL(unittest.TestCase):
    directory = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.directory = os.path.join(directory, '../../files/visualization/base/')
        cls.directory_blobs = os.path.join(directory, '../../files/visualization/blobs/')
        cls.directory_svc = os.path.join(directory, '../../files/visualization/svc/')
        cls.directory_germancredit = os.path.join(directory, '../../files/visualization/germancredit/')

    def test_init_should_init(self):
        vw = mls.sl.visualize.VisualizeLogSL(directory=self.directory)
        self.assertIsInstance(vw.context, mls.sl.models.Context)
        self.assertIsNone(vw.figure)
        self.assertIsNone(vw.scoreText)
        self.assertIsNone(vw.confusionMatrixFigure)
        self.assertIsNone(vw.data_test_table)
        self.assertIsNone(vw.fairness_results)

    def test_task_load_data_data_loaded(self):
        vw = mls.sl.visualize.VisualizeLogSL(directory=self.directory)
        vw.task_load_data()
        self.assertEqual(100, len(vw.context.data.x))
        self.assertEqual(100, len(vw.context.data.y))
        self.assertEqual(20, len(vw.context.data_test.x))
        self.assertEqual(20, len(vw.context.data_test.y))
        self.assertEqual(80, len(vw.context.data_train.x))
        self.assertEqual(80, len(vw.context.data_train.y))
        self.assertEqual(2, vw.context.algorithm.hyperparameters['n_neighbors'])
        self.assertEqual('sklearn.neighbors.KNeighborsClassifier', vw.context.algorithm.algorithm_family)
        self.assertIsInstance(vw.context.classifier, neighbors.KNeighborsClassifier)
        self.assertEqual(1.00, vw.context.evaluation.score)
        np.testing.assert_array_equal(np.array([[13, 0], [0, 7]]), vw.context.evaluation.confusion_matrix)

    def test_task_load_data_with_fairness_data_loaded(self):
        vw = mls.sl.visualize.VisualizeLogSL(directory=self.directory_germancredit)
        vw.task_load_data()
        self.assertAlmostEqual(-0.12854990969960323, vw.context.evaluation.sub_evaluation.demographic_parity)

    def test_task_display_data_config_generated(self):
        vw = mls.sl.visualize.VisualizeLogSL(directory=self.directory)
        vw.task_load_data()
        vw.task_display_data()
        self.assertIsInstance(vw.figure, dcc.Graph)
        self.assertIsInstance(vw.scoreText, html.P)
        self.assertIsInstance(vw.confusionMatrixFigure, dcc.Graph)
        self.assertIsInstance(vw.data_test_table, dash_table.DataTable)

    def test_task_display_data_blobs_figure_generated(self):
        vw = mls.sl.visualize.VisualizeLogSL(directory=self.directory_blobs)
        vw.task_load_data()
        vw.task_display_data()
        self.assertIsInstance(vw.figure, dcc.Graph)
        self.assertIsInstance(vw.scoreText, html.P)
        self.assertIsInstance(vw.confusionMatrixFigure, dcc.Graph)
        self.assertIsInstance(vw.data_test_table, dash_table.DataTable)

    def test_task_display_data_svc_figure_generated(self):
        vw = mls.sl.visualize.VisualizeLogSL(directory=self.directory_svc)
        vw.task_load_data()
        vw.task_display_data()
        self.assertIsInstance(vw.figure, dcc.Graph)
        self.assertIsInstance(vw.scoreText, html.P)
        self.assertIsInstance(vw.confusionMatrixFigure, dcc.Graph)
        self.assertIsInstance(vw.data_test_table, dash_table.DataTable)

    def test_task_display_data_fairness_div_generated(self):
        vw = mls.sl.visualize.VisualizeLogSL(directory=self.directory_germancredit)
        vw.task_load_data()
        vw.task_display_data()
        self.assertIsInstance(vw.fairness_results, html.Div)
        self.assertIsNotNone(vw.fairness_results.children)

    def test_task_display_data_no_fairness_div_generated_empty(self):
        vw = mls.sl.visualize.VisualizeLogSL(directory=self.directory)
        vw.task_load_data()
        vw.task_display_data()
        self.assertIsInstance(vw.fairness_results, html.Div)
        self.assertIsNone(vw.fairness_results.children)

    def test_task_display_data_more_2_dimensions(self):
        """
        :test : mlsurvey.workflows.VisualizationWorkflow.task_display_data()
        :condition : dataset has more than 2 dimensions
        :main_result : figure is not defined
        """
        vw = mls.sl.visualize.VisualizeLogSL(directory=self.directory_germancredit)
        vw.task_load_data()
        vw.task_display_data()
        self.assertIsNone(vw.figure)
        self.assertIsInstance(vw.scoreText, html.P)
        self.assertIsInstance(vw.confusionMatrixFigure, dcc.Graph)
        self.assertIsInstance(vw.data_test_table, dash_table.DataTable)

    def test_run_all_step_should_be_executed(self):
        vw = mls.sl.visualize.VisualizeLogSL(directory=self.directory)
        vw.run()
        self.assertIsInstance(vw.scoreText, html.P)

    def test_result_should_return_visualized(self):
        vw = mls.sl.visualize.VisualizeLogSL(directory=self.directory)
        vw.run()
        parameters = {'display_config': 'block', 'display_figure': 'block', 'display_data_test_table': True}
        result = vw.get_result(parameters)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], html.Div)
        self.assertEqual(2, len(result))

import os
import unittest

import dash_html_components as html

import mlsurvey as mls


class TestVisualizeLogDetail(unittest.TestCase):
    directory = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.directory = os.path.join(directory, '../files/visualization/base/')
        cls.directory_blobs = os.path.join(directory, '../files/visualization/blobs/')
        cls.directory_svc = os.path.join(directory, '../files/visualization/svc/')
        cls.directory_germancredit = os.path.join(directory, '../files/visualization/germancredit/')

    def test_init_should_init(self):
        vw = mls.visualize.VisualizeLogDetail(directory=self.directory)
        self.assertEqual(self.directory, vw.source_directory)
        self.assertIsNone(vw.config)
        self.assertIsInstance(vw.log, mls.Logging)
        self.assertEqual(self.directory, vw.log.directory)
        self.assertIsNone(vw.configText)

    def test_task_load_data_data_loaded(self):
        vw = mls.visualize.VisualizeLogDetail(directory=self.directory)
        vw.task_load_data()
        self.assertTrue('DataSet1', vw.config.data['learning_process']['parameters']['input'])

    def test_task_display_data_config_generated(self):
        vw = mls.visualize.VisualizeLogDetail(directory=self.directory)
        vw.task_load_data()
        vw.task_display_data()
        self.assertIsInstance(vw.configText, html.Div)

    def test_run_all_step_should_be_executed(self):
        vw = mls.visualize.VisualizeLogDetail(directory=self.directory)
        vw.run()
        self.assertIsInstance(vw.configText, html.Div)

    def test_result_should_return_visualized(self):
        vw = mls.visualize.VisualizeLogDetail(directory=self.directory)
        vw.run()
        parameters = {'display_config': 'block'}
        result = vw.get_result(parameters)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], html.Div)

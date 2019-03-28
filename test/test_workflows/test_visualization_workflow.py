import os
import shutil
import unittest

from bokeh.models.widgets import Paragraph, Div
from bokeh.plotting import Figure
from sklearn import neighbors

import mlsurvey as mls


class TestVisualizationWorkflow(unittest.TestCase):
    directory = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.directory = os.path.join(directory, '../files/visualization/')

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir)
        os.remove(os.path.join(cls.directory, 'result.html'))

    def test_init_should_init(self):
        vw = mls.workflows.VisualizationWorkflow(directory=self.directory)
        self.assertEqual(vw.source_directory, self.directory)
        self.assertIsInstance(vw.slw, mls.workflows.SupervisedLearningWorkflow)
        self.assertIsNone(vw.figure)
        self.assertIsNone(vw.scoreText)
        self.assertIsNone(vw.configText)

    def test_set_terminated_all_terminated(self):
        vw = mls.workflows.VisualizationWorkflow(directory=self.directory)
        vw.task_terminated_load_data = True
        vw.task_terminated_display_data = True
        vw.set_terminated()
        self.assertTrue(vw.terminated)

    def test_set_terminated_all_terminated_but_load_data(self):
        vw = mls.workflows.VisualizationWorkflow(directory=self.directory)
        vw.task_terminated_load_data = False
        vw.task_terminated_display_data = True
        vw.set_terminated()
        self.assertFalse(vw.terminated)

    def test_set_terminated_all_terminated_but_display_data(self):
        vw = mls.workflows.VisualizationWorkflow(directory=self.directory)
        vw.task_terminated_load_data = True
        vw.task_terminated_display_data = False
        vw.set_terminated()
        self.assertFalse(vw.terminated)

    def test_task_load_data_data_loaded(self):
        vw = mls.workflows.VisualizationWorkflow(directory=self.directory)
        vw.task_load_data()
        self.assertTrue('DataSet1', vw.slw.config.data['learning_process']['input'])
        self.assertEqual(100, len(vw.slw.context.data.x))
        self.assertEqual(100, len(vw.slw.context.data.y))
        self.assertEqual(20, len(vw.slw.context.data_test.x))
        self.assertEqual(20, len(vw.slw.context.data_test.y))
        self.assertEqual(80, len(vw.slw.context.data_train.x))
        self.assertEqual(80, len(vw.slw.context.data_train.y))
        self.assertIsInstance(vw.slw.context.classifier, neighbors.KNeighborsClassifier)
        self.assertEqual(1.00, vw.slw.context.evaluation.score)

    def test_task_display_data_figure_generated(self):
        vw = mls.workflows.VisualizationWorkflow(directory=self.directory)
        vw.task_load_data()
        vw.task_display_data()
        self.assertIsInstance(vw.figure, Figure)
        self.assertIsInstance(vw.scoreText, Paragraph)
        self.assertIsInstance(vw.configText, Div)
        self.assertTrue(os.path.isfile(vw.slw.log.directory + 'result.html'))

    def test_run_all_step_should_be_executed(self):
        vw = mls.workflows.VisualizationWorkflow(directory=self.directory)
        self.assertFalse(vw.terminated)
        vw.run()

        # data is loaded
        self.assertTrue(vw.task_terminated_load_data)
        # figure is defined
        self.assertTrue(vw.task_terminated_display_data)

        # all tasks are finished
        self.assertTrue(vw.terminated)

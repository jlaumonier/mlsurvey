import json

from bokeh.io import output_file
from bokeh.layouts import row
from bokeh.models.widgets import Paragraph, Div
from bokeh.plotting import figure, save
from json2html import *

import mlsurvey as mls
from .learning_workflow import LearningWorkflow


class VisualizationWorkflow(LearningWorkflow):
    """ Workflow for visualizing learning results """

    def __init__(self, directory):
        """
        initialize the workflow reading the files from directory
        :param directory: the directory where the results are stored
        """
        super().__init__()
        self.source_directory = directory
        self.slw = mls.SupervisedLearningWorkflow()
        self.task_terminated_load_data = False
        self.task_terminated_display_data = False
        self.figure = None
        self.scoreText = None
        self.configText = None

    def set_terminated(self):
        self.terminated = (self.task_terminated_load_data
                           & self.task_terminated_display_data)

    def task_load_data(self):
        self.slw.load_data_classifier(self.source_directory)
        self.task_terminated_load_data = True

    def task_display_data(self):
        x = self.slw.data_train.x
        color_list = ['red', 'green', 'blue']
        colors = [color_list[y] for y in self.slw.data_train.y]
        self.figure = figure()
        self.figure.circle(x[:, 0], x[:, 1], color=colors, fill_alpha=0.2, size=10)
        self.scoreText = Paragraph(text="""Score : """ + str(self.slw.score))
        self.configText = Div(text=json2html.convert(json.dumps(self.slw.config.data)))
        page_layout = row(children=[self.configText, self.figure, self.scoreText], sizing_mode='stretch_both')
        output_file(self.slw.log.directory + 'result.html', title='Result of learning')
        save(page_layout)
        self.task_terminated_display_data = True

    def run(self):
        self.task_load_data()
        self.task_display_data()
        self.set_terminated()

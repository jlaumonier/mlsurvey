import json

import colorcet as cc
import numpy as np
from bokeh.io import output_file
from bokeh.layouts import row
from bokeh.models import LinearColorMapper, ColorBar, AdaptiveTicker
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
        self.slw = mls.workflows.SupervisedLearningWorkflow()
        self.task_terminated_load_data = False
        self.task_terminated_display_data = False
        self.figure = None
        self.scoreText = None
        self.configText = None

    def set_terminated(self):
        self.terminated = (self.task_terminated_load_data
                           & self.task_terminated_display_data)

    def task_load_data(self):
        """
        Load data, config, classifier and statistic from directory
        """
        self.slw.load_data_classifier(self.source_directory)
        self.task_terminated_load_data = True

    def task_display_data(self):
        """
        Display with bokeh. This methode is garbage for testing and MUST be refactored
        """

        x = self.slw.context.data.x
        x_train = self.slw.context.data_train.x
        x_test = self.slw.context.data_test.x
        xx, yy = mls.Utils.make_meshgrid(x[:, 0], x[:, 1])
        color_list = ['#0000FF', '#FF0000']
        # colors = [color_list[y] for y in self.slw.data.y]
        colors_train = [color_list[y] for y in self.slw.context.data_train.y]
        colors_test = [color_list[y] for y in self.slw.context.data_test.y]
        self.figure = figure(x_range=(xx.min(), xx.max()), y_range=(yy.min(), yy.max()))

        self.scoreText = Paragraph(text="""Score : """ + str(self.slw.context.score))
        self.configText = Div(text=json2html.convert(json.dumps(self.slw.config.data)))

        if hasattr(self.slw.context.classifier, "decision_function"):
            z = self.slw.context.classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            z = self.slw.context.classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        color_mapper = LinearColorMapper(
            palette=cc.rainbow,
            low=z.min(),
            high=z.max()
        )
        z = z.reshape(xx.shape)
        dh = yy.max() - yy.min()
        dw = xx.max() - xx.min()
        self.figure.image(image=[z], color_mapper=color_mapper,
                          dh=[dh], dw=[dw], x=[xx.min()], y=[yy.min()], global_alpha=0.5)

        self.figure.circle(x_train[:, 0], x_train[:, 1], color=colors_train, fill_alpha=1.0, size=10)
        self.figure.cross(x_test[:, 0], x_test[:, 1], color=colors_test, fill_alpha=1.0, size=10)

        color_bar = ColorBar(color_mapper=color_mapper, ticker=AdaptiveTicker(),
                             label_standoff=12, border_line_color=None, location=(0, 0))

        self.figure.add_layout(color_bar, 'right')
        page_layout = row(children=[self.configText, self.figure, self.scoreText], sizing_mode='stretch_both')
        output_file(self.slw.log.directory + 'result.html', title='Result of learning')
        save(page_layout)
        self.task_terminated_display_data = True

    def run(self):
        self.task_load_data()
        self.task_display_data()
        self.set_terminated()

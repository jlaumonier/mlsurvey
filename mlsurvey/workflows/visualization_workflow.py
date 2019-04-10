import colorlover as cl
import dash_core_components as dcc
import dash_dangerously_set_inner_html as ddsih
import dash_html_components as html
import json2table
import numpy as np
import plotly.graph_objs as go

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
        self.config = None
        self.context = mls.models.Context(eval_type=mls.models.EvaluationSupervised)
        self.log = mls.Logging(self.source_directory, base_dir='')
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
        self.config = mls.Config('config.json', self.source_directory)
        self.context.load(self.log)
        self.task_terminated_load_data = True

    def task_display_data(self):
        """
        Display with dash.
        """

        x = self.context.data.x
        x_train = self.context.data_train.x
        y_train = self.context.data_train.y
        x_test = self.context.data_test.x
        y_test = self.context.data_test.y
        mesh_step = .3
        xx, yy = mls.Utils.make_meshgrid(x[:, 0], x[:, 1], mesh_step)

        bright_cscale = [[0, '#FF0000'], [1, '#0000FF']]

        colorscale_zip = zip(np.arange(0, 1.01, 1 / 8),
                             cl.scales['9']['div']['RdBu'])
        cscale = list(map(list, colorscale_zip))

        if hasattr(self.context.classifier, "decision_function"):
            z = self.context.classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            z = self.context.classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        scaled_threshold = 0.5 * (z.max() - z.min()) + z.min()
        r = max(abs(scaled_threshold - z.min()),
                abs(scaled_threshold - z.max()))

        layout = go.Layout(
            xaxis=dict(
                ticks='',
                showticklabels=False,
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                ticks='',
                showticklabels=False,
                showgrid=False,
                zeroline=False,
            ),
            hovermode='closest',
            legend=dict(x=0, y=-0.01, orientation="h"),
            margin=dict(l=0, r=0, t=0, b=0),
        )

        trace0 = go.Contour(
            x=np.arange(xx.min(), xx.max(), mesh_step),
            y=np.arange(yy.min(), yy.max(), mesh_step),
            z=z.reshape(xx.shape),
            zmin=scaled_threshold - r,
            zmax=scaled_threshold + r,
            hoverinfo='none',
            showscale=False,
            contours=dict(
                showlines=False
            ),
            colorscale=cscale,
            opacity=0.9
        )

        # Plot the threshold
        trace1 = go.Contour(
            x=np.arange(xx.min(), xx.max(), mesh_step),
            y=np.arange(yy.min(), yy.max(), mesh_step),
            z=z.reshape(xx.shape),
            showscale=False,
            hoverinfo='none',
            contours=dict(
                showlines=False,
                type='constraint',
                operation='=',
                value=scaled_threshold,
            ),
            name=f'Threshold ({scaled_threshold:.3f})',
            line=dict(
                color='#222222'
            )
        )

        trace2 = go.Scatter(
            x=x_train[:, 0],
            y=x_train[:, 1],
            mode='markers',
            name=f'Training Data)',
            marker=dict(
                size=10,
                color=y_train,
                colorscale=bright_cscale,
                line=dict(
                    width=1
                )
            )
        )

        trace3 = go.Scatter(
            x=x_test[:, 0],
            y=x_test[:, 1],
            mode='markers',
            name=f'Test Data (accuracy={self.context.evaluation.score:.3f})',
            marker=dict(
                size=10,
                symbol='cross',
                color=y_test,
                colorscale=bright_cscale,
                line=dict(
                    width=1
                ),
            )
        )
        data = [trace0, trace1, trace2, trace3]
        f = go.Figure(data=data, layout=layout)
        self.figure = dcc.Graph(id='graph', figure=f)
        self.scoreText = html.P('Score : ' + str(self.context.evaluation.score))
        # This line makes a cannot find reference warning and i do not know why and how i can fix it
        self.configText = html.Div([ddsih.DangerouslySetInnerHTML(json2table.convert(self.config.data))])
        self.task_terminated_display_data = True

    def run(self):
        self.task_load_data()
        self.task_display_data()
        self.set_terminated()

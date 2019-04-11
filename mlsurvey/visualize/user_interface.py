import webbrowser

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import mlsurvey as mls


class UserInterface:

    def __init__(self, directory):
        self.analyse_logs = mls.visualize.AnalyzeLogs(directory)
        self.analyse_logs.store_config()

    @staticmethod
    def update(value):
        vw = mls.workflows.VisualizationWorkflow(value)
        vw.run()
        result = [html.Div(vw.configText, className='six columns'),
                  html.Div(vw.figure, className='four columns'),
                  html.Div(vw.scoreText, className='two columns')]
        return result

    def run(self):
        app = dash.Dash(__name__)

        options = [{'label': self.analyse_logs.list_dir[idx], 'value': self.analyse_logs.list_full_dir[idx] + '/'}
                   for idx in range(len(self.analyse_logs.list_dir))]

        app.layout = html.Div(children=[
            html.H1(children='ML Survey'),
            html.Div(children='''
                 Visualization of results
             '''),
            html.Div(children=[dcc.Dropdown(
                id='dir-choice-id',
                options=options,
                value=options[1]['value'],
                className='two columns'
            ),
                dcc.Loading(
                    id="loading",
                    children=[html.Div(id='visualize-id',
                                       className='ten columns')],
                    type="default",
                    className='row')
            ],
                className='row')
        ])

        app.callback(
            Output(component_id='visualize-id', component_property='children'),
            [Input(component_id='dir-choice-id', component_property='value')])(self.update)

        webbrowser.open_new('localhost:8050')
        app.run_server(debug=False)

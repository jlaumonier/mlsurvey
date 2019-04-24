import dash_core_components as dcc
import dash_html_components as html
import dash_table
import tinydb as tdb
from dash.dependencies import Input, Output

import mlsurvey as mls


class SearchInterface:

    def __init__(self, analyse_logs):
        self.analyse_logs = analyse_logs

    def search(self, n_clicks, value_algo, value_ds):
        """Callback when the user click on the 'Search' button"""
        result = []
        if n_clicks is not None:
            query = tdb.Query()
            search_result = self.analyse_logs.db.search(
                query.learning_process.algorithm['algorithm-family'].matches(value_algo)
                & query.learning_process.input.type.matches(value_ds))
            for res in search_result:
                one_row = {'Algorithm': res['learning_process']['algorithm']['algorithm-family'],
                           'AlgoParams': str(res['learning_process']['algorithm']['hyperparameters']),
                           'Dataset': res['learning_process']['input']['type'],
                           'DSParams': str(res['learning_process']['input']['parameters']),
                           'Directory': res['location']}
                result.append(one_row)
        return result

    @staticmethod
    def select_result(derived_virtual_data, derived_virtual_selected_rows):
        """Callback when the user select one row in the result table"""
        result = []
        if derived_virtual_selected_rows is not None and len(derived_virtual_selected_rows) != 0:
            result = []
            for idx in derived_virtual_selected_rows:
                directory = derived_virtual_data[idx]['Directory']
                vw = mls.workflows.VisualizationWorkflow(directory)
                vw.run()
                if vw.figure is None:
                    one_result = html.Div(children=[html.Div(vw.configText, className='six columns'),
                                                    html.Div(vw.scoreText, className='six columns')],
                                          className='one_result')
                else:
                    one_result = html.Div(children=[html.Div(vw.configText, className='six columns'),
                                                    html.Div(vw.figure, className='four columns'),
                                                    html.Div(vw.scoreText, className='two columns')],
                                          className='one_result')
                result.append(one_result)
        return result

    def get_layout(self):
        """Layout of the search page"""
        d = [{'Algorithm': 0, 'AlgoParams': 0, 'Dataset': 0, 'DSParams': 0, 'Directory': 0}]
        c = ['Algorithm', 'AlgoParams', 'Dataset', 'DSParams', 'Directory']
        options_algorithms = [{'label': a, 'value': a} for a in self.analyse_logs.algorithms_list]
        options_datasets = [{'label': d, 'value': d} for d in self.analyse_logs.datasets_list]
        return html.Div(children=[
            dcc.Dropdown(id='search-algo-dd-id',
                         options=options_algorithms,
                         className='three columns',
                         value=options_algorithms[0]['value'],
                         searchable=False,
                         clearable=False,
                         placeholder="Algorithm"),
            dcc.Dropdown(id='search-dataset-dd-id',
                         options=options_datasets,
                         className='three columns',
                         value=options_datasets[0]['value'],
                         searchable=False,
                         clearable=False,
                         placeholder="Dataset"
                         ),
            html.Button('Search', id='search-button-id'),
            dash_table.DataTable(id='search-results-id',
                                 columns=[{"name": i, "id": i} for i in c],
                                 data=d,
                                 row_selectable='multi',
                                 pagination_mode='fe',
                                 pagination_settings={
                                     "displayed_pages": 1,
                                     "current_page": 0,
                                     "page_size": 10,
                                 },
                                 navigation="page",
                                 selected_rows=[],
                                 style_as_list_view=True,
                                 style_cell={'textAlign': 'left', 'font-size': '0.9em'})], className='twelve columns')

    def define_callback(self, dash_app):
        """define the callbacks on the page"""
        dash_app.callback(
            Output(component_id='search-results-id', component_property='data'),
            [Input(component_id='search-button-id', component_property='n_clicks'),
             Input(component_id='search-algo-dd-id', component_property='value'),
             Input(component_id='search-dataset-dd-id', component_property='value')])(self.search)

        dash_app.callback(
            Output(component_id='visualize-id',
                   component_property='children'),
            [Input(component_id='search-results-id',
                   component_property='derived_virtual_data'),
             Input(component_id='search-results-id',
                   component_property='derived_virtual_selected_rows')])(self.select_result)

import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output

import mlsurvey as mls


class SearchInterface:

    def __init__(self, analyse_logs):
        self.analyse_logs = analyse_logs

    def search(self, n_clicks):
        """Callback when the user click on the 'Search' button"""
        result = []
        if n_clicks is not None:
            search_result = self.analyse_logs.db.all()
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
                one_result = html.Div(children=[html.Div(vw.configText, className='six columns'),
                                                html.Div(vw.figure, className='four columns'),
                                                html.Div(vw.scoreText, className='two columns')])
                result.append(one_result)
        return result

    @staticmethod
    def get_layout():
        """Layout of the search page"""
        d = [{'Algorithm': 0, 'AlgoParams': 0, 'Dataset': 0, 'DSParams': 0, 'Directory': 0}]
        c = ['Algorithm', 'AlgoParams', 'Dataset', 'DSParams', 'Directory']
        return html.Div(children=[
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
                                 style_cell={'textAlign': 'left', 'font-size': '0.9em'})],
            className='twelve columns')

    def define_callback(self, dash_app):
        """define the callbacks on the page"""
        dash_app.callback(
            Output(component_id='search-results-id', component_property='data'),
            [Input(component_id='search-button-id', component_property='n_clicks')])(self.search)

        dash_app.callback(
            Output(component_id='visualize-id',
                   component_property='children'),
            [Input(component_id='search-results-id',
                   component_property='derived_virtual_data'),
             Input(component_id='search-results-id',
                   component_property='derived_virtual_selected_rows')])(self.select_result)

from functools import reduce
from operator import eq, ge, gt, le, lt, ne

import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import tinydb as tdb
from dash.dependencies import Input, Output, State

import mlsurvey as mls


class SearchInterface:

    def __init__(self, analyse_logs):
        self.analyse_logs = analyse_logs

    def search(self, n_clicks, value_algo, value_ds, criteria_values):
        """Callback when the user click on the 'Search' button"""
        """ helped by on https://stackoverflow.com/questions/30530562/dynamically-parse-and-build-tinydb-queries"""
        operators = {'==': eq, '!=': ne, '<=': le, '>=': ge, '<': lt, '>': gt}
        result = []
        queries = []
        db = self.analyse_logs.db
        query = tdb.Query()
        if criteria_values:
            for c in criteria_values:
                dict_criteria = eval(c)
                criteria = dict_criteria['criteria']
                criteria_operator = dict_criteria['criteria_operator']
                criteria_value = dict_criteria['criteria_value']
                criteria_split = criteria.split('.')
                q = reduce(tdb.Query.__getitem__, criteria_split, query.learning_process)
                operator = operators[criteria_operator]
                dt = self.analyse_logs.parameters_df.dtypes[criteria]
                if dt == np.float64:
                    criteria_value = np.float64(criteria_value)
                if dt == np.int64:
                    criteria_value = np.int64(criteria_value)
                one_query = operator(q, criteria_value)
                queries.append(one_query)
        if queries:
            search_result = db.search(reduce(lambda a, b: a & b, queries)
                                      & query.learning_process.algorithm['algorithm-family'].matches(value_algo)
                                      & query.learning_process.input.type.matches(value_ds)
                                      )
        else:
            search_result = db.search(query.learning_process.algorithm['algorithm-family'].matches(value_algo)
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
    def add_criteria(n_clicks_timestamp, criteria_value, criteria_operator, criteria, existing_options,
                     existing_values):
        if existing_values is None:
            existing_values = []
        if criteria is not None and criteria_value is not None and criteria_operator is not None:
            label = criteria + criteria_operator + criteria_value
            value = {
                'criteria': criteria,
                'criteria_operator': criteria_operator,
                'criteria_value': criteria_value
            }
            existing_options.append({'label': label, 'value': str(value)})
            existing_values.append(str(value))
        return existing_options, existing_values

    def select_criteria_value_operator(self, criteria):
        result_value = None
        result_operator = None
        if criteria:
            values = self.analyse_logs.parameters_df[criteria].unique()
            result_value = [{'label': str(v), 'value': str(v)} for v in values]
            dtype = self.analyse_logs.parameters_df.dtypes[criteria]
            if dtype == np.float64 or dtype == np.int64:
                operators = ['==', '!=', '<=', '>=', '<', '>']
            else:
                operators = ['==', '!=']
            result_operator = [{'label': str(v), 'value': str(v)} for v in operators]
        return result_value, result_operator

    @staticmethod
    def select_result(derived_virtual_data, derived_virtual_selected_rows, options):
        """Callback when the user select one row in the result table"""
        result = []
        display_config = 'block' if 'CFG' in options else 'none'
        display_figure = 'block' if 'FIG' in options else 'none'
        display_data_test_table = True if 'DTA_TEST_TBL' in options else False
        if derived_virtual_selected_rows is not None and len(derived_virtual_selected_rows) != 0:
            for idx in derived_virtual_selected_rows:
                directory = derived_virtual_data[idx]['Directory']
                vw = mls.workflows.VisualizationWorkflow(directory)
                vw.run()

                data_test_section = html.Details(children=[html.Summary('Test Data'),
                                                           vw.data_test_table])
                evaluation_result = html.Div(children=[html.Div(vw.scoreText),
                                                       html.Div(vw.confusionMatrixFigure),
                                                       vw.fairness_results])
                if vw.figure is None:
                    one_result = html.Div(children=[html.Div(vw.configText,
                                                             className='six columns',
                                                             style={'display': display_config}),
                                                    html.Div(evaluation_result, className='six columns')],
                                          className='one_result')
                else:
                    one_result = html.Div(children=[html.Div(vw.configText,
                                                             className='five columns',
                                                             style={'display': display_config}),
                                                    html.Div(vw.figure,
                                                             className='three columns',
                                                             style={'display': display_figure}),
                                                    html.Div(evaluation_result, className='four columns')],
                                          className='one_result')
                result.append(one_result)

                if display_data_test_table:
                    result.append(data_test_section)

                result.append(html.Hr())
        return result

    def get_layout(self):
        """Layout of the search page"""
        d = [{'Algorithm': 0, 'AlgoParams': 0, 'Dataset': 0, 'DSParams': 0, 'Directory': 0}]
        col_names = ['Algorithm', 'AlgoParams', 'Dataset', 'DSParams', 'Directory']
        options_algorithms = [{'label': a, 'value': a} for a in self.analyse_logs.algorithms_list]
        options_datasets = [{'label': d, 'value': d} for d in self.analyse_logs.datasets_list]

        list_criteria = [{'label': a, 'value': a} for a in self.analyse_logs.parameters_df.columns]

        crit_drop = [dcc.Dropdown(id='id-criteria',
                                  options=list_criteria,
                                  className='three columns',
                                  value=list_criteria[0]['value'],
                                  searchable=False,
                                  clearable=False,
                                  placeholder='Criteria'),
                     dcc.Dropdown(id='id-criteria-operator',
                                  options=[],
                                  className='one column',
                                  searchable=False,
                                  clearable=False,
                                  placeholder='Operator'),
                     dcc.Dropdown(id='id-criteria-value',
                                  options=[],
                                  className='three columns',
                                  searchable=False,
                                  clearable=False,
                                  placeholder='Value'),
                     html.Button('Add criteria', id='add-crit-button-id'),
                     dcc.Dropdown(id='id-criteria-list',
                                  options=[],
                                  className='seven columns',
                                  searchable=False,
                                  clearable=True,
                                  multi=True)
                     ]

        list_search_section_children = [dcc.Dropdown(id='search-algo-dd-id',
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
                                                             columns=[{"name": i, "id": i} for i in col_names],
                                                             data=d,
                                                             row_selectable='multi',
                                                             page_action='native',
                                                             page_size=10,
                                                             page_current=0,
                                                             selected_rows=[],
                                                             style_as_list_view=True,
                                                             style_cell={'textAlign': 'left',
                                                                         'font-size': '0.9em'})]
        criteria_section = html.Div(id='criteria-section',
                                    children=crit_drop,
                                    className='twelve columns')

        search_section = html.Div(id='search-section',
                                  children=list_search_section_children,
                                  className='twelve columns')

        options_section = html.Div(id='option-section',
                                   children=[
                                       dcc.Checklist(
                                           id='options-checklist',
                                           options=[
                                               {'label': 'Configuration table', 'value': 'CFG'},
                                               {'label': 'Data Separation figure', 'value': 'FIG'},
                                               {'label': 'Data test table', 'value': 'DTA_TEST_TBL'}],
                                           value=['CFG', 'FIG'],
                                           labelStyle={'display': 'inline-block'})
                                   ],
                                   className='twelve columns')

        criteria_detail = html.Details(children=[html.Summary('Criteria'), criteria_section],
                                       open=True)
        search_detail = html.Details(children=[html.Summary('Search'), search_section],
                                     open=True)
        options_detail = html.Details(children=[html.Summary('Display options'), options_section],
                                      open=False)
        return html.Div(children=[criteria_detail, search_detail, options_detail])

    def define_callback(self, dash_app):
        """define the callbacks on the page"""
        dash_app.callback(
            Output(component_id='search-results-id', component_property='data'),
            [Input(component_id='search-button-id', component_property='n_clicks'),
             Input(component_id='search-algo-dd-id', component_property='value'),
             Input(component_id='search-dataset-dd-id', component_property='value'),
             Input(component_id='id-criteria-list', component_property='value')])(self.search)

        dash_app.callback(
            Output(component_id='visualize-id',
                   component_property='children'),
            [Input(component_id='search-results-id',
                   component_property='derived_virtual_data'),
             Input(component_id='search-results-id',
                   component_property='derived_virtual_selected_rows'),
             Input(component_id='options-checklist',
                   component_property='value')])(self.select_result)

        dash_app.callback(
            [Output(component_id='id-criteria-value',
                    component_property='options'),
             Output(component_id='id-criteria-operator',
                    component_property='options')],
            [Input(component_id='id-criteria',
                   component_property='value')])(self.select_criteria_value_operator)

        dash_app.callback(
            [Output(component_id='id-criteria-list',
                    component_property='options'),
             Output(component_id='id-criteria-list',
                    component_property='value')],
            [Input(component_id='add-crit-button-id',
                   component_property='n_clicks_timestamp')],
            [State(component_id='id-criteria-value',
                   component_property='value'),
             State(component_id='id-criteria-operator',
                   component_property='value'),
             State(component_id='id-criteria',
                   component_property='value'),
             State(component_id='id-criteria-list',
                   component_property='options'),
             State(component_id='id-criteria-list',
                   component_property='value')])(self.add_criteria)

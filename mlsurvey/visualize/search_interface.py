from functools import reduce
from operator import eq, ge, gt, le, lt, ne

import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import tinydb as tdb
from dash.dependencies import Input, Output, State

import mlsurvey as mls


class SearchInterface:

    def __init__(self, analyse_logs):
        self.analyse_logs = analyse_logs
        self.col_names = []
        for e in self.analyse_logs.lists:
            self.col_names.append(e)
            self.col_names.append(e + 'Params')
        self.col_names.append('FairnessParams')
        self.col_names.append('Directory')
        self.col_names.append('Type')

    @staticmethod
    def get_result_figure_summary(search_result):
        """
        get the result of the summary
        :param search_result: result to filter
        :return: sorted_df: dataframe containing the data for axis x (first columns) and for axis y (second column)
                            sorted by axis x
                 list_of_not_unique_key: List of parameters which does not have a unique value.
        """
        sorted_df = pd.DataFrame({'x': [], 'y': []})
        flatten_results = [mls.Utils.flatten_dict(doc['learning_process']['parameters'], separator='.') for doc in
                           search_result]
        result_df = pd.DataFrame(flatten_results)
        nb_value = {}
        values = {}
        # find the possible values for each column
        for col, val in result_df.items():
            values[col] = result_df[col].unique().tolist()
            nb_value[col] = len(values[col])
        # list of the key that have more than one value
        list_of_not_unique_key = [key for (key, value) in nb_value.items() if value > 1]
        # if there is only one key (for 2d figures)
        if len(list_of_not_unique_key) == 1:
            # get the values of the key that have more than 1 value
            x = values[list_of_not_unique_key[0]]
            y = []
            # get the index of result
            idx_result = result_df.index[result_df[list_of_not_unique_key[0]] == x].tolist()
            for i in idx_result:
                # for each directory, launch a visualization workflow to get the evaluation
                vw_class = mls.Utils.import_from_dotted_path(search_result[i]['learning_process']['type'])
                vw = vw_class.visualize_class()(search_result[i]['location'])
                vw.run()
                # append the score to the y axis
                # TODO TEMPORARY (hum !) PATCH
                if hasattr(vw, 'context'):
                    y.append(vw.context.evaluation.score)
                if hasattr(vw, 'evaluation'):
                    y.append(vw.evaluation.score)
            # in order to sort the result by 'x', we create a dataframe
            df = pd.DataFrame({list_of_not_unique_key[0]: x, 'score': y})
            # and we sort it.
            sorted_df = df.sort_values(by=[list_of_not_unique_key[0]])
        return sorted_df, list_of_not_unique_key

    def search(self, value_algo, value_ds, criteria_values):
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
                q = reduce(tdb.Query.__getitem__, criteria_split, query.learning_process.parameters)
                operator = operators[criteria_operator]
                dt = self.analyse_logs.parameters_df.dtypes[criteria]
                if dt == np.float64:
                    criteria_value = np.float64(criteria_value)
                if dt == np.int64:
                    criteria_value = np.int64(criteria_value)
                if dt == bool:
                    criteria_value = mls.Utils.str2bool(criteria_value)
                one_query = operator(q, criteria_value)
                queries.append(one_query)
        if queries:
            search_result = db.search(reduce(lambda a, b: a & b, queries)
                                      & query.learning_process.parameters.input.type.matches(value_ds)
                                      )
        else:
            search_result = db.search(query.learning_process.parameters.input.type.matches(value_ds))
        for res in search_result:
            if 'fairness' in res['learning_process']['parameters']['input']:
                str_fairness_params = str(res['learning_process']['parameters']['input']['fairness'])
            else:
                str_fairness_params = str({})
            one_row = {}
            for e in self.analyse_logs.lists:
                # exclude some lists
                if e != 'image_files' and e != 'json_files':
                    one_row[e] = res['learning_process']['parameters'][e]['type']
                    eparam = 'parameters'
                    if 'hyperparameters' in res['learning_process']['parameters'][e]:
                        eparam = 'hyperparameters'
                    one_row[e + 'Params'] = str(res['learning_process']['parameters'][e][eparam])
            one_row['FairnessParams'] = str_fairness_params
            one_row['Directory'] = res['location']
            one_row['Type'] = res['learning_process']['type']
            result.append(one_row)
        return result, search_result

    def search_callback(self, n_clicks, value_algo, value_ds, criteria_values):
        """Callback when the user click on the 'Search' button"""
        result, search_result = self.search(value_algo, value_ds, criteria_values)
        result_df, list_of_not_unique_key = self.get_result_figure_summary(search_result)
        figure = dcc.Graph(id='graph-id')
        html_list_not_unique = html.Ul([])
        for k in list_of_not_unique_key:
            html_list_not_unique.children.append(html.Li(k, style={'margin-bottom': '0em'}))
            if len(result_df) != 0 and 'score' in result_df.columns:
                data = {'x': result_df[result_df.columns[0]], 'y': result_df['score']}
                f = go.Figure(data=data,
                              layout=go.Layout(
                                  xaxis={'title': str(result_df.columns[0])},
                                  yaxis={'title': 'score'}
                              )
                              )
                figure = dcc.Graph(id='graph-summary-id', figure=f)
        div_summary = [html.Div(children=html_list_not_unique,
                                className='three columns'),
                       html.Div(children=figure,
                                className='five columns')]
        return result, div_summary

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
    def select_result(derived_virtual_data, derived_virtual_selected_rows, options, images, json):
        """Callback when the user select one row in the result table"""
        result = []
        parameters = {'display_config': 'block' if 'CFG' in options else 'none',
                      'display_figure': 'block' if 'FIG' in options else 'none',
                      'display_data_test_table': True if 'DTA_TEST_TBL' in options else False,
                      'images': images if images else [],
                      'json': json if json else []}
        if derived_virtual_selected_rows is not None and len(derived_virtual_selected_rows) != 0:
            for idx in derived_virtual_selected_rows:
                vw_class = mls.Utils.import_from_dotted_path(derived_virtual_data[idx]['Type'])
                vw = vw_class.visualize_class()(derived_virtual_data[idx]['Directory'])
                vw.run()
                one_result = vw.get_result(parameters)
                result.extend(one_result)
                result.append(html.Hr())
        return result

    def get_layout(self):
        """Layout of the search page"""
        empty_list = [dict.fromkeys(self.col_names, 0)]
        options_algorithms = [{'label': '.', 'value': '.'}]
        if 'algorithm' in self.analyse_logs.lists:
            options_algorithms = [{'label': a, 'value': a} for a in self.analyse_logs.lists['algorithm']]
        options_datasets = [{'label': d, 'value': d} for d in self.analyse_logs.lists['input']]

        options_images = [{'label': d, 'value': d} for d in self.analyse_logs.lists['image_files']]

        options_json = [{'label': d, 'value': d} for d in self.analyse_logs.lists['json_files']]

        list_criteria = [{'label': a, 'value': a} for a in sorted(self.analyse_logs.parameters_df.columns)]

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
                                                             columns=[{"name": i, "id": i} for i in self.col_names],
                                                             data=empty_list,
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
                                           labelStyle={'display': 'inline-block'}),
                                       dcc.Dropdown(id='options-images',
                                                    options=options_images,
                                                    className='five columns',
                                                    searchable=False,
                                                    clearable=False,
                                                    placeholder="Images",
                                                    multi=True
                                                    ),
                                       dcc.Dropdown(id='options-json',
                                                    options=options_json,
                                                    className='five columns',
                                                    searchable=False,
                                                    clearable=False,
                                                    placeholder="JSON",
                                                    multi=True
                                                    ),
                                   ],
                                   className='twelve columns')

        figure_summary_section = html.Div(id='figure-summary-id',
                                          className='twelve columns')

        criteria_detail = html.Details(children=[html.Summary('Criteria'), criteria_section],
                                       open=True)
        search_detail = html.Details(children=[html.Summary('Search'), search_section],
                                     open=True)
        options_detail = html.Details(children=[html.Summary('Display options'), options_section],
                                      open=False)
        figure_summary_details = html.Details(children=[html.Summary('Figure summary'), figure_summary_section],
                                              open=False)
        return html.Div(children=[criteria_detail, search_detail, options_detail, figure_summary_details])

    def define_callback(self, dash_app):
        """define the callbacks on the page"""
        dash_app.callback(
            [Output(component_id='search-results-id', component_property='data'),
             Output(component_id='figure-summary-id', component_property='children')],
            [Input(component_id='search-button-id', component_property='n_clicks'),
             Input(component_id='search-algo-dd-id', component_property='value'),
             Input(component_id='search-dataset-dd-id', component_property='value'),
             Input(component_id='id-criteria-list', component_property='value')])(self.search_callback)

        dash_app.callback(
            Output(component_id='visualize-id',
                   component_property='children'),
            [Input(component_id='search-results-id',
                   component_property='derived_virtual_data'),
             Input(component_id='search-results-id',
                   component_property='derived_virtual_selected_rows'),
             Input(component_id='options-checklist',
                   component_property='value'),
             Input(component_id='options-images',
                   component_property='value'),
             Input(component_id='options-json',
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

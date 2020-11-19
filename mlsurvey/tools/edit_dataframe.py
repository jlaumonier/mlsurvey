import dash
from dash.dependencies import Input, Output, State
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import os


def load_data(sheet_name):
    df = dict_df[sheet_name]
    local_lc = [{"name": str(i), "id": str(i)} for i in df.columns.tolist()]
    local_data_values = df.to_dict(orient='records')
    width = ['5%', '5%', '5%', '5%', '5%', '5%', '5%', '5%']
    local_style_cell_conditional = []
    for i, c in enumerate(df.columns.tolist()):
        if i < len(width):
            local_style_cell_conditional.append({'if': {'column_id': c}, 'width': width[i]})
    return local_lc, local_data_values, local_style_cell_conditional


path = '/home/julien/prog/CCF-transformation-numerique-competences/data/'
filename = "Analyse de bases de données_Offre d'emploi des compagnies d'assurance.xlsx"

dict_df = pd.read_excel(os.path.join(path, filename), sheet_name=None)
sheets_list = list(dict_df.keys())
sheets_options = [{'label': s, 'value': s} for s in sheets_list]

app = dash.Dash(__name__)

style_cell = {'overflow': 'hidden',
              'textOverflow': 'ellipsis',
              'maxWidth': 0,
              'whiteSpace': 'normal',
              'max-height': '50px'
              }

lc, data_values, style_cell_conditional = load_data(sheets_list[0])

app.layout = html.Div([
    dcc.Dropdown(
        id='sheet-dropdown',
        options=sheets_options,
        value=sheets_list[0]
    ),
    dash_table.DataTable(
        id='table-editing-simple',
        style_cell=style_cell,
        columns=lc,
        data=data_values,
        style_cell_conditional=style_cell_conditional,
        page_size=2,
        editable=False,
        row_deletable=False
    ),

    html.Button('Edit', id='edit-button', n_clicks=0),
    # html.Div([
    #     dcc.Input(
    #         id='editing-columns-name',
    #         placeholder='Enter a column name...',
    #         value='',
    #         style={'padding': 10}
    #     ),
    #     html.Button('Add Column', id='editing-columns-button', n_clicks=0)
    # ], style={'height': 50}),
    html.Button('Save', id='save-button', n_clicks=0),
    html.P('Not saved', id='save-state'),
    dcc.Textarea(
        id='textarea-example',
        value='Textarea content initialized\nwith multiple lines of text',
        style={'width': '100%', 'height': 400},
    )
])


@app.callback(
    [Output('table-editing-simple', 'columns'),
     Output('table-editing-simple', 'data'),
     Output('table-editing-simple', 'style_cell_conditional')],
    Input('sheet-dropdown', 'value'))
def change_sheet(value):
    return load_data(value)


@app.callback(
    Output('save-state', 'children'),
    [Input('save-button', 'n_clicks'),
     Input('table-editing-simple', 'data')])
def save(n_clicks, d):
    if n_clicks > 0:
        save_df = pd.DataFrame(d)
        save_df.to_excel(os.path.join(path, 'test.xlsx'), engine='xlsxwriter', index=False)
        return 'Saved'
    else:
        return 'Not Saved'


# @app.callback(
#     [Output('table-editing-simple', 'columns'),
#      Output('table-editing-simple', 'style_cell_conditional')],
#     [Input('editing-columns-button', 'n_clicks')],
#     [State('editing-columns-name', 'value'),
#      State('table-editing-simple', 'columns'),
#      State('table-editing-simple', 'style_cell_conditional')])
# def update_columns(n_clicks, value, existing_columns, cell_style):
#     if n_clicks > 0:
#         existing_columns.append({
#             'id': value, 'name': value,
#             'renamable': True, 'deletable': True
#         })
#         cell_style.append({'if': {'column_id': value}, 'width': '10%'})
#     return existing_columns, cell_style

@app.callback(
    Output('textarea-example', 'value'),
    Input('edit-button', 'n_clicks'),
    [State('table-editing-simple', 'active_cell'),
     State('table-editing-simple', 'data')])
def edit_cell(n_clicks, active_cell, data):
    if n_clicks > 0 and active_cell:
        row = active_cell['row']
        col = active_cell['column']
        col_id = active_cell['column_id']
        if col_id in data[int(row)]:
            val = data[int(row)][col_id]
        else:
            val = None
        return str(val)
    else:
        return 'Select a cell to edit...'


if __name__ == '__main__':
    app.run_server(debug=True)

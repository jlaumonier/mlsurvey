import dash_core_components as dcc
import dash_html_components as html


class DetailInterface:

    @staticmethod
    def get_layout():
        result = html.Div(children=[
            html.Div(children='''
                 Visualization of results
             '''),
            dcc.Loading(
                id="loading",
                children=[html.Div(id='visualize-id',
                                   className='twelve columns')],
                type="default")
        ])

        return result
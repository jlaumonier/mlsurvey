import dash_html_components as html


class DetailInterface:

    @staticmethod
    def get_layout():
        result = html.Div(children=[
            html.H2(children='''
                         Results
                     '''),
            html.Div(id='visualize-id',
                     className='twelve columns')
        ])

        return result

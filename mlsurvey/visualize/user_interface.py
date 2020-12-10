import dash
import dash_html_components as html

import mlsurvey as mls


class UserInterface:

    def __init__(self, directory):
        self.analyse_logs = mls.visualize.AnalyzeLogs(directory)
        self.analyse_logs.store_config()
        self.search_interface = mls.visualize.SearchInterface(self.analyse_logs)
        self.detail_interface = mls.visualize.DetailInterface()

    def run(self):
        app = dash.Dash(__name__)

        app.scripts.config.serve_locally = True

        app.layout = html.Div(children=[html.H1(children='Machine Learning Survey'),
                                        self.search_interface.get_layout(),
                                        self.detail_interface.get_layout()])
        self.search_interface.define_callback(app)

        app.run_server(debug=False, host='0.0.0.0')

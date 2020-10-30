import dash_dangerously_set_inner_html as ddsih
import dash_html_components as html
import json2table

import mlsurvey as mls


class VisualizeLogDetail:
    """ Generation of the visualization for a log experiment """

    def __init__(self, directory):
        """
        initialize the workflow reading the files from directory
        :param directory: the directory where the results are stored
        """
        super().__init__()
        self.source_directory = directory
        self.config = None
        self.configText = None
        self.log = mls.Logging(self.source_directory, base_dir='')

    def task_load_data(self):
        """
        Load config from directory
        """
        self.config = mls.Config('config.json', self.source_directory)

    def task_display_data(self):
        """
        Display with dash.
        """
        self.config.compact()
        # This line makes a cannot find reference warning and i do not know why and how i can fix it
        self.configText = html.Div([ddsih.DangerouslySetInnerHTML(json2table.convert(self.config.data))])

    def run(self):
        self.task_load_data()
        self.task_display_data()

    def get_result(self, parameters):
        result = html.Div(children=[html.Div(self.configText,
                                             className='six columns',
                                             style={'display': parameters['display_config']})],
                          className='one_result')
        return [result]

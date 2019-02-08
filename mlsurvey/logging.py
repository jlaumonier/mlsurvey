import datetime
import json
import os

import mlsurvey as mls


class Logging:

    def __init__(self, dir_name=None):
        self.base_dir = 'logs/'
        if dir_name is None:
            dir_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.directory = self.base_dir + dir_name + '/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def save_input(self, inpt: mls.Input):
        output = {'input.x': inpt.x.tolist(), 'input.y': inpt.y.tolist()}
        with open(self.directory + 'input.json', 'w') as outfile:
            json.dump(output, outfile)

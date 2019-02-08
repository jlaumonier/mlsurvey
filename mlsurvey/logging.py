import datetime
import json
import os

import mlsurvey as mls


class Logging:

    def __init__(self, dir_name=None, base_dir='logs/'):
        self.base_dir = base_dir
        if dir_name is None:
            dir_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.directory = self.base_dir + dir_name + '/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def save_input(self, inpt: mls.Input):
        output = inpt.to_dict()
        with open(self.directory + 'input.json', 'w') as outfile:
            json.dump(output, outfile)

    def load_input(self, filename):
        with open(self.directory + filename, 'r') as infile:
            data = json.load(infile)
        i = mls.Input()
        i.from_dict(data)
        return i

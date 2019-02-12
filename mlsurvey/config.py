import json

from dotmap import DotMap


class Config:

    def __init__(self, name='config.json', directory='config/'):
        """
        load a json config file named 'name' from config/ directory. self.data is a DotMap
        :param name: name of the config file
        """
        with open(directory + name, 'r') as json_config_file:
            self.data = DotMap(json.load(json_config_file))

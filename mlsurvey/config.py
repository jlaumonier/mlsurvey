import json

from dotmap import DotMap


class Config:

    def __init__(self, name='config.json'):
        """
        load a json config file named 'name' from config/ directory. self.data is a DotMap
        :param name: name of the config file
        """
        with open('config/' + name, 'r') as json_config_file:
            self.data = DotMap(json.load(json_config_file))

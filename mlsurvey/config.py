import json


class Config:

    def __init__(self, name='config.json', directory='config/', config=None):
        """
        load a json config file named 'name' from config/ directory. self.data is a dictionary
        :param name: name of the config file
        :param directory: directory of the config file
        :param config; dictionary that contain the config. The file is not loaded if config is set
        """
        if config is None:
            with open(directory + name, 'r') as json_config_file:
                self.data = json.load(json_config_file)
        else:
            self.data = config

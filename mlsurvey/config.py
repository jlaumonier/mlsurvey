import json
import os

import mlsurvey as mls


class Config:

    def __init__(self, name='config.json', directory='config/', config=None):
        """
        load a json config file named 'name' from config/ directory. self.data is a dictionary
        :param name: name of the config file
        :param directory: directory of the config file
        :param config; dictionary that contain the config. The file is not loaded if config is set
        Raise FileNotFoundError if the config file does not exists
        Raise ConfigError is the config file is not a valid Json file
        """
        if config is None:
            full_path = os.path.join(directory, name)
            with open(full_path, 'r') as json_config_file:
                try:
                    self.data = json.load(json_config_file)
                except json.JSONDecodeError as e:
                    raise mls.exceptions.ConfigError(e)
        else:
            self.data = config

    @staticmethod
    def compact(config):
        """
        compact a config dictionary to a compact config dictionary
        compact config does not include definition but only learning process. Does not support Fairness process.
        :return: compact config
        """
        dataset_name = config['learning_process']['input']
        dataset_dict = config['datasets'][dataset_name]
        split_name = config['learning_process']['split']
        split_dict = config['splits'][split_name]
        algorithm_name = config['learning_process']['algorithm']
        algorithm_dict = config['algorithms'][algorithm_name]

        compact_config = {'learning_process': {}}
        compact_config['learning_process']['input'] = dataset_dict
        compact_config['learning_process']['split'] = split_dict
        compact_config['learning_process']['algorithm'] = algorithm_dict

        return compact_config

import json

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
            try:
                self.data = mls.FileOperation.load_json_as_dict(name, directory)
            except json.JSONDecodeError as e:
                raise mls.exceptions.ConfigError(e)
        else:
            self.data = config

    @property
    def data(self):
        return self.__data__

    @data.setter
    def data(self, d):
        """
        setter to insure self.data is always python-ready
        """
        self.__data__ = mls.Utils.transform_to_dict(d)

    def is_compacted(self):
        return '#refs' not in self.data

    def compact(self):
        """
        compact a config dictionary to a compact config dictionary. Replace ref ('@') by their values in '#refs'
        exclude all refs in the result. DO NOT HANDLE LOOPS !
        Modify current config
        """

        def _change_string(string: str, refs: dict):
            ref_result = string
            if string[0] == '@':
                path = string[1:].split('.')
                value_path = refs
                for p in path:
                    value_path = value_path[p]
                ref_result = value_path
            return ref_result

        def _update_ref(sub_config, refs):
            sub_result = sub_config
            for k, v in sub_config.items():
                if isinstance(v, dict):
                    _update_ref(v, refs)
                if isinstance(v, str):
                    sub_result[k] = _change_string(v, refs)
                if isinstance(v, list):
                    for i, list_element in enumerate(v):
                        if isinstance(list_element, str):
                            sub_result[k][i] = _change_string(list_element, refs)
                        else:
                            sub_result[k][i] = list_element
            return sub_result

        if not self.is_compacted():
            self.data = _update_ref(self.data, self.data['#refs'])
            del self.data['#refs']


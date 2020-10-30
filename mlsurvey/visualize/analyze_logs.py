import os
from operator import concat

import pandas as pd
import tinydb as tdb

import mlsurvey as mls


class AnalyzeLogs:

    def __init__(self, directory):
        self.directory = directory
        list_dir = sorted(os.listdir(self.directory))
        self.list_full_dir = list(map(concat, [self.directory] * len(list_dir), list_dir))
        self.list_full_dir = list(filter(AnalyzeLogs._log_dir_is_not_multiple_learning_result, self.list_full_dir))
        self.lists = {}
        self.parameters_df = None
        self.db = tdb.TinyDB(storage=tdb.storages.MemoryStorage)

    def __del__(self):
        self.db.close()

    @staticmethod
    def _log_dir_is_not_multiple_learning_result(directory):
        return not os.path.isfile(os.path.join(directory, 'results.json'))

    def store_config(self):
        """
        Read all config.json files and store them into a db
        """
        for i, d in enumerate(self.list_full_dir):
            config_json = mls.FileOperation.load_json_as_dict('config.json', d, tuple_to_string=True)
            config = mls.Config(config=config_json)
            config.compact()
            config.data['location'] = d
            self.db.insert(config.data)
            print('\rAnalyze logs..'+str(int((i+1)*100/len(self.list_full_dir)))+'%', end='')
        self.fill_lists()

    def fill_lists(self):
        """
        Fill lists from config 'learning_process', 'parameters'
        """
        all_doc = self.db.all()
        for doc in all_doc:
            for param in doc['learning_process']['parameters']:
                t = [doc['learning_process']['parameters'][param]['type'] for doc in all_doc]
                self.lists[param] = list(set(t))
                self.lists[param].sort()
                self.lists[param].insert(0, '.')
        parameters_list = [mls.Utils.flatten_dict(doc['learning_process']['parameters'], separator='.') for doc in
                           all_doc]
        self.parameters_df = pd.DataFrame(parameters_list)

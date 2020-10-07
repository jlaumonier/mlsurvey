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
        self.algorithms_list = []
        self.datasets_list = []
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
        for d in self.list_full_dir:
            config_json = mls.FileOperation.load_json_as_dict('config.json', d, tuple_to_string=True)
            config = mls.Config(config=config_json)
            config.compact()
            config.data['location'] = d
            self.db.insert(config.data)
        self.fill_lists()

    def fill_lists(self):
        """
        Fill algorithms_list and datasets_list
        """
        all_doc = self.db.all()
        a = [doc['learning_process']['parameters']['algorithm']['algorithm-family'] for doc in all_doc]
        self.algorithms_list = list(set(a))
        self.algorithms_list.sort()
        self.algorithms_list.insert(0, '.')
        d = [doc['learning_process']['parameters']['input']['type'] for doc in all_doc]
        self.datasets_list = list(set(d))
        self.datasets_list.sort()
        self.datasets_list.insert(0, '.')
        parameters_list = [mls.Utils.flatten_dict(doc['learning_process']['parameters'], separator='.') for doc in
                           all_doc]
        self.parameters_df = pd.DataFrame(parameters_list)

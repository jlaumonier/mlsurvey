import os
from operator import concat

import pandas as pd
import tinydb as tdb

import mlsurvey as mls


class AnalyzeLogs:

    def __init__(self, directory):
        self.directory = directory
        self.list_dir = sorted(os.listdir(self.directory))
        self.list_full_dir = list(map(concat, [self.directory] * len(self.list_dir), self.list_dir))
        self.algorithms_list = []
        self.datasets_list = []
        self.parameters_df = None
        self.db = tdb.TinyDB(storage=tdb.storages.MemoryStorage)

    def __del__(self):
        self.db.close()

    def store_config(self):
        """
        Read all config.json files and store them into a db
        """
        for d in self.list_full_dir:
            log = mls.Logging(dir_name=d, base_dir='')
            config = log.load_json_as_dict('config.json')
            compact_config = mls.Config.compact(config)
            compact_config['location'] = d
            self.db.insert(compact_config)
        self.fill_lists()

    def fill_lists(self):
        """
        Fill algorithms_list and datasets_list
        """
        all_doc = self.db.all()
        a = [doc['learning_process']['algorithm']['algorithm-family'] for doc in all_doc]
        self.algorithms_list = list(set(a))
        self.algorithms_list.sort()
        self.algorithms_list.insert(0, '.')
        d = [doc['learning_process']['input']['type'] for doc in all_doc]
        self.datasets_list = list(set(d))
        self.datasets_list.sort()
        self.datasets_list.insert(0, '.')
        parameters_list = [mls.Utils.flatten_dict(doc['learning_process'], separator='.') for doc in all_doc]
        self.parameters_df = pd.DataFrame(parameters_list)
        print(self.parameters_df.to_string())

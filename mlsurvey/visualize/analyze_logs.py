import os
from operator import concat

import tinydb as tdb

import mlsurvey as mls


class AnalyzeLogs:

    def __init__(self, directory):
        self.directory = directory
        self.list_dir = sorted(os.listdir(self.directory))
        self.list_full_dir = list(map(concat, [self.directory] * len(self.list_dir), self.list_dir))
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

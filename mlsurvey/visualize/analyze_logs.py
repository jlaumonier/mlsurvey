import os
from operator import concat


class AnalyzeLogs:

    def __init__(self, directory):
        self.directory = directory
        self.list_dir = sorted(os.listdir(self.directory))
        self.list_full_dir = list(map(concat, [self.directory] * len(self.list_dir), self.list_dir))

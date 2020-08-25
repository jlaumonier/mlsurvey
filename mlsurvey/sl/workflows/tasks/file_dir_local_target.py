import luigi
import os


class FileDirLocalTarget(luigi.LocalTarget):

    def __init__(self, filename, directory, format_=None, is_tmp=False):
        super().__init__(path=os.path.join(directory, filename), format=format_, is_tmp=is_tmp)
        self.directory = directory
        self.filename = filename

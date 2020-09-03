import luigi
import os

import mlsurvey as mls


class BaseTask(luigi.Task):

    base_directory = luigi.Parameter(default='')
    logging_base_directory = luigi.Parameter()
    logging_directory = luigi.Parameter()
    config_directory = luigi.Parameter(default='config/')
    config_filename = luigi.Parameter(default='config.json')
    log = None
    config = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # init log and config
        self.init_log_config()

    def init_log_config(self):
        """
        Initialized log and config
        """
        # TODO refactoring directory
        final_config_directory = os.path.join(str(self.base_directory), str(self.config_directory))
        self.log = mls.Logging(dir_name=self.logging_directory, base_dir=self.logging_base_directory)
        self.config = mls.Config(name=self.config_filename, directory=final_config_directory)


import luigi

import mlsurvey as mls


class SupervisedLearningTask(luigi.Task):

    logging_base_directory = luigi.Parameter()
    logging_directory = luigi.Parameter()
    config_directory = luigi.Parameter()
    config_filename = luigi.Parameter()
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
        self.log = mls.Logging(dir_name=self.logging_directory, base_dir=self.logging_base_directory)
        self.config = mls.Config(name=self.config_filename, directory=self.config_directory)
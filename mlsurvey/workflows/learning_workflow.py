from abc import abstractmethod
import mlsurvey as mls


class LearningWorkflow:

    def __init__(self, config_file='config.json',
                 config_directory='config/',
                 base_directory='',
                 logging_dir=None):
        """
        :param config_file config file name, default config.json
        :param config_directory
        :param base_directory
        :param logging_dir
        """
        self.terminated = False
        self.config_directory = config_directory
        self.base_directory = base_directory
        self.config_file = config_file
        self.log = mls.Logging(dir_name=logging_dir)
        self.task_terminated_init = True

    @abstractmethod
    def run(self):
        pass

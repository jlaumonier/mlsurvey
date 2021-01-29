from abc import abstractmethod
import logging
import os
import mlsurvey as mls


class LearningWorkflow:

    def __init__(self, config_file='config.json',
                 config_directory='config/',
                 base_directory='',
                 logging_dir=None,
                 mlflow_log=False, mlflow_tracking_uri=None, mlflow_xp_name='Default'):
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
        # init mlflow
        final_config_directory = os.path.join(str(base_directory), str(config_directory))
        self.config = mls.Config(name=config_file, directory=final_config_directory)
        self.config.compact()
        self.log = mls.Logging(base_dir=base_directory,
                               dir_name=logging_dir,
                               mlflow_log=mlflow_log, mlflow_tracking_uri=mlflow_tracking_uri,
                               mlflow_xp_name=mlflow_xp_name)
        self.task_terminated_init = True

    def terminate(self):
        self.log.msg('Workflow ' + self.__class__.__name__ + ' is terminated', logging.INFO)
        self.log.terminate_mlflow()
        self.terminated = True
        dict_terminated = {'Terminated': self.terminated}
        self.log.save_dict_as_json('terminated.json', dict_terminated)

    @staticmethod
    def visualize_class():
        return mls.visualize.VisualizeLogDetail

    @abstractmethod
    def run(self):
        pass

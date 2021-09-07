from abc import abstractmethod
import logging
import os

from codecarbon.emissions_tracker import EmissionsTracker

import mlsurvey as mls


class LearningWorkflow:
    sources_directories = None

    def __init__(self, config_file='config.json',
                 config_directory='config/',
                 config_dict=None,
                 base_directory='',
                 logging_dir=None,
                 mlflow_log=False, mlflow_xp_name=None,
                 sources_directories=None):
        """
        :param config_file config file name, default config.json
        :param config_directory
        :param base_directory
        :param logging_dir
        :param mlflow_log
        :param mlflow_xp_name
        :poram sources_directories
        """
        self.terminated = False
        self.config_directory = config_directory
        self.base_directory = base_directory
        self.config_file = config_file
        # init mlflow
        final_config_directory = os.path.join(str(base_directory), str(config_directory))
        if config_dict:
            self.config = mls.Config(config=config_dict)
        else:
            self.config = mls.Config(name=config_file, directory=final_config_directory)
        self.config.compact()
        mlflow_tracking_uri = None
        if 'mlflow' in self.config.app_config:
            mlflow_tracking_uri = self.config.app_config['mlflow']['tracking_uri']
        self.log = mls.Logging(base_dir=os.path.join(base_directory, 'logs'),
                               dir_name=logging_dir,
                               mlflow_log=mlflow_log, mlflow_tracking_uri=mlflow_tracking_uri,
                               mlflow_xp_name=mlflow_xp_name)
        self.emissions_tracker = EmissionsTracker(output_dir=self.log.directory)
        self.emissions_tracker.start()
        self.task_terminated_init = True
        self.sources_directories = sources_directories
        if self.sources_directories:
            print('Copying sources directories...', end='')
            for src_dir_k in sources_directories.keys():
                self.log.copy_source_tree(source=self.sources_directories[src_dir_k], dest_dir=src_dir_k)
            print('Done')

    def terminate(self):
        self.log.msg('Workflow ' + self.__class__.__name__ + ' is terminated', logging.INFO)
        url = self.log.terminate_mlflow()
        self.terminated = True
        dict_terminated = {'Terminated': self.terminated, 'mlflow_run_url': url}
        self.log.save_dict_as_json('terminated.json', dict_terminated)
        self.emissions_tracker.stop()
        self.log.save_mlflow_artifact('emissions.csv')

    @staticmethod
    def visualize_class():
        return mls.visualize.VisualizeLogDetail

    @abstractmethod
    def run(self):
        pass

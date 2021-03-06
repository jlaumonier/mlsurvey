from abc import abstractmethod
import logging
import mlflow
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
        # init mlflow
        client = mlflow.tracking.MlflowClient()
        experiments = client.list_experiments()
        run = client.create_run(experiments[0].experiment_id)
        self.log = mls.Logging(dir_name=logging_dir, mlflow_run_id=run.info.run_id)
        self.task_terminated_init = True

    def terminate(self):
        self.log.msg('Workflow ' + self.__class__.__name__ + ' is terminated', logging.INFO)
        self.log.mlflow_client.set_terminated(self.log.mlflow_run_id)
        self.terminated = True
        dict_terminated = {'Terminated': self.terminated}
        self.log.save_dict_as_json('terminated.json', dict_terminated)

    @staticmethod
    def visualize_class():
        return mls.visualize.VisualizeLogDetail

    @abstractmethod
    def run(self):
        pass

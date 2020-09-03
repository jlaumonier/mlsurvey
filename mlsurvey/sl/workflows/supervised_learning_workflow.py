import os

import luigi

import mlsurvey as mls
from mlsurvey.workflows.learning_workflow import LearningWorkflow


class SupervisedLearningWorkflow(LearningWorkflow):

    def __init__(self,
                 config_file='config.json',
                 config=None,
                 config_directory='config/',
                 base_directory='',
                 logging_dir=None):
        """
        Initialized the supervised learning workflow
        :param config_file: config file for initializing the workflow, Used if config is None
        :param config: dictionary for config. If set, replace config file
        """
        super().__init__(config_directory=config_directory, base_directory=base_directory)
        self.config_file = config_file
        self.log = mls.Logging(dir_name=logging_dir)

    def run(self):
        """
        Run all tasks
        """
        luigi.build([mls.sl.workflows.tasks.EvaluateTask(logging_directory=self.log.dir_name,
                                                         logging_base_directory=os.path.join(self.base_directory,
                                                                                             self.log.base_dir),
                                                         config_filename=self.config_file,
                                                         config_directory=self.config_directory,
                                                         base_directory=self.base_directory)],
                    local_scheduler=True)

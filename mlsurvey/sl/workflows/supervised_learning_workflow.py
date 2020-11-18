import os

import luigi

import mlsurvey as mls
from mlsurvey.workflows.learning_workflow import LearningWorkflow


class SupervisedLearningWorkflow(LearningWorkflow):

    @staticmethod
    def visualize_class():
        return mls.sl.visualize.VisualizeLogSL

    def run(self):
        """
        Run all tasks
        """
        luigi.build([mls.sl.workflows.tasks.EvaluateTask(logging_directory=self.log.dir_name,
                                                         logging_base_directory=os.path.join(self.base_directory,
                                                                                             self.log.base_dir),
                                                         config_filename=self.config_file,
                                                         config_directory=self.config_directory,
                                                         base_directory=self.base_directory,
                                                         mlflow_run_id=self.log.mlflow_run_id)],
                    local_scheduler=True)
        self.terminate()

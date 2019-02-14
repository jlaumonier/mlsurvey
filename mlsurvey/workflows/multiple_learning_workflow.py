from dotmap import DotMap

import mlsurvey as mls
from .learning_workflow import LearningWorkflow


class MultipleLearningWorkflow(LearningWorkflow):

    def __init__(self, config_file='config.json'):
        super().__init__()
        self.task_terminated_expand_config = False
        self.task_terminated_run_each_config = False
        self.config = mls.Config(config_file)
        self.expanded_config = []
        self.slw = []

    def set_terminated(self):
        self.terminated = self.task_terminated_expand_config \
                          & self.task_terminated_run_each_config

    def task_expand_config(self):
        for lp_element in mls.Utils.dict_generator_cartesian_product(self.config.data.learning_process.toDict()):
            one_config = {'datasets': self.config.data.datasets.toDict(),
                          'algorithms': self.config.data.algorithms.toDict(),
                          'splits': self.config.data.splits.toDict(),
                          'learning_process': lp_element}
            self.expanded_config.append(DotMap(one_config))
        self.task_terminated_expand_config = True

    def task_run_each_config(self):
        for c in self.expanded_config:
            sl = mls.SupervisedLearningWorkflow(config=c.toDict())
            sl.run()
            self.slw.append(sl)
        self.task_terminated_run_each_config = True

    def run(self):
        self.task_expand_config()
        self.task_run_each_config()
        self.set_terminated()


import mlsurvey as mls
from .learning_workflow import LearningWorkflow


class MultipleLearningWorkflow(LearningWorkflow):

    def __init__(self, config_file='config.json'):
        super().__init__()
        self.config = mls.Config(config_file)

    def set_terminated(self):
        self.terminated = True

    def run(self):
        self.set_terminated()


from abc import abstractmethod


class LearningWorkflow:

    def __init__(self, config_directory='config/'):
        self.terminated = False
        self.config_directory = config_directory
        self.task_terminated_init = True

    @abstractmethod
    def set_terminated(self):
        pass

    @abstractmethod
    def run(self):
        pass


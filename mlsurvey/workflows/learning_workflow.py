from abc import abstractmethod


class LearningWorkflow:

    def __init__(self, config_directory='config/'):
        self.terminated = False
        self.config_directory = config_directory

    @abstractmethod
    def set_terminated(self):
        pass

    @abstractmethod
    def run(self):
        pass


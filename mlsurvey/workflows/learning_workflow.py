from abc import abstractmethod


class LearningWorkflow:

    def __init__(self):
        self.terminated = False

    @abstractmethod
    def set_terminated(self):
        pass

    @abstractmethod
    def run(self):
        pass


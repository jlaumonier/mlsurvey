from abc import abstractmethod


class DataSet:

    def __init__(self):
        self.x = []
        self.y = []

    @abstractmethod
    def generate(self):
        pass

    class Factory:
        @abstractmethod
        def create(self): pass

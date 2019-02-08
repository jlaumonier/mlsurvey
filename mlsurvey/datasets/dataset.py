from abc import abstractmethod


class DataSet:

    def __init__(self):
        self.x = []
        self.y = []
        self.params = {}

    def set_generation_parameters(self, params):
        """
        set the parameters of the generation
        :param params:
        """
        self.params = params

    @abstractmethod
    def generate(self):
        pass

    class Factory:
        @abstractmethod
        def create(self): pass

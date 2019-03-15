from abc import abstractmethod


class DataSet:

    def __init__(self, t):
        self.x = []
        self.y = []
        self.params = {}
        self.t = t

    def set_generation_parameters(self, params):
        """
        set the parameters of the generation
        :param params: a dictionary containing parameters. Empty dictionary or None if no parameter
        """
        if params is None:
            params = {}
        self.params = params

    @abstractmethod
    def generate(self):
        pass

    class Factory:
        @staticmethod
        @abstractmethod
        def create(t): pass

from abc import abstractmethod


class DataSet:

    def __init__(self, t):
        """
        initialize the dataset
        :param t: type of the dataset (usually the name of the class or the name of the function called)
        """
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

    def to_dict(self):
        """
        transform the dataset into a dictionary {'type': .., 'parameters': {...} }
        :return: the dictionary
        """
        result = {'type': self.t, 'parameters': self.params}
        return result

    @abstractmethod
    def generate(self):
        pass

    class Factory:
        @staticmethod
        @abstractmethod
        def create(t): pass

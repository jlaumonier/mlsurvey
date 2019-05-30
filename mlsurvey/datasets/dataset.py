from abc import abstractmethod

import mlsurvey as mls


class DataSet:

    def __init__(self, t):
        """
        initialize the dataset
        :param t: type of the dataset (usually the name of the class or the name of the function called)
        """
        self.params = {}
        self.t = t
        self.fairness = {}

    def set_generation_parameters(self, params):
        """
        set the parameters of the generation
        :param params: a dictionary containing parameters. Empty dictionary or None if no parameter
        """
        if params is None:
            params = {}
        self.params = params

    def set_fairness_parameters(self, fairness):
        """
        set the fairness parameter of the dataset
        :param fairness: fairness dictionary containing protected_attribute
        """
        if fairness != {} \
                and fairness is not None \
                and 'protected_attribute' in fairness \
                and 'privileged_classes' in fairness:
            self.fairness = fairness
        else:
            raise mls.exceptions.ConfigError('Fairness parameter not valid')

    def to_dict(self):
        """
        transform the dataset into a dictionary {'type': .., 'parameters': {...} }
        :return: the dictionary
        """
        result = {'type': self.t, 'parameters': self.params}
        if self.fairness != {}:
            result['fairness'] = self.fairness
        return result

    @abstractmethod
    def generate(self):
        pass

    class Factory:
        @staticmethod
        @abstractmethod
        def create(t): pass

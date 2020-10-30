import mlsurvey as mls


class Algorithm:

    def __init__(self, config):
        """
        Initialize the algorithm class using the config
        :param config dictionary containing keys 'type' and 'hyperparamters'.
        The config 'type' is the name of the class defining the algorithm (e.g. sklearn.svm.SVC)
        the config 'hyperparamters' is a dictionary used to initialize the class
        Raise a mlsurvey.exceptions.ConfigError if keys are not found in config
        """
        self.algorithm_family = None
        self.hyperparameters = None
        try:
            self.algorithm_family = config['type']
            self.hyperparameters = config['hyperparameters']
        except KeyError as e:
            raise mls.exceptions.ConfigError(e)

    def learn(self, x, y):
        """learn a classifier from input x and y"""
        try:
            classifier_class = mls.Utils.import_from_dotted_path(self.algorithm_family)
        except AttributeError as e:
            raise mls.exceptions.ConfigError(e)
        classifier = classifier_class(**self.hyperparameters)
        classifier.fit(x, y)
        return classifier

    def to_dict(self):
        """
        transform an algorithm to a dictionary  {'type': ..., 'hyperparameters': ...}
        :return: a dictionary
        """
        result = {'type': self.algorithm_family, 'hyperparameters': self.hyperparameters}
        return result

    @staticmethod
    def from_dict(d):
        """
        Create an Algorithm from a dictionary
        :param d: the dictionary
        :return: an instance of Algorithm
        """
        return Algorithm(d)

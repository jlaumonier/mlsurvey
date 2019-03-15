import mlsurvey as mls


class Algorithm:

    def __init__(self, config):
        """Initialize the algorithm class"""
        self.algorithm_family = config['algorithm-family']
        self.hyperparameters = config['hyperparameters']

    def learn(self, x, y):
        """learn a classifier from input x and y"""
        classifier_class = mls.Utils.import_from_dotted_path(self.algorithm_family)
        classifier = classifier_class(**self.hyperparameters)
        classifier.fit(x, y)
        return classifier

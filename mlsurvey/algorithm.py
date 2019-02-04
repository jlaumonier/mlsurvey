from sklearn import neighbors


class Algorithm:

    def __init__(self, af, h):
        """Initialize the algorithm class"""
        self.algorithm_family = af
        self.hyperparameters = h

    def learn(self, x, y):
        """learn a classifier from input x and y"""
        classifier = neighbors.KNeighborsClassifier(self.hyperparameters['n_neighbors'],
                                                    algorithm=self.hyperparameters['algorithm'],
                                                    weights=self.hyperparameters['weights']
                                                    )
        classifier.fit(x, y)
        return classifier

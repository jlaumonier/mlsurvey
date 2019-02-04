from sklearn import datasets


class Input:

    def __init__(self):
        iris = datasets.load_iris()
        # we only take the first two features. We could avoid this ugly
        # slicing by using a two-dim dataset
        self.x = iris.data[:, :2]
        self.y = iris.target

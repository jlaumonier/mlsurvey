from sklearn import datasets

from .dataset import DataSet
from .dataset_factory import DataSetFactory


class Iris(DataSet):

    def generate(self):
        iris = datasets.load_iris()
        # we only take the first two features. We could avoid this ugly
        # slicing by using a two-dim dataset
        self.x = iris.data[:, :2]
        self.y = iris.target

    class Factory:
        @staticmethod
        def create(): return Iris()


DataSetFactory.add_factory('Iris', Iris.Factory)

from sklearn.datasets import make_classification

from .dataset import DataSet
from .dataset_factory import DataSetFactory


class NClassRandomClassification(DataSet):

    def generate(self):
        self.x, self.y = make_classification(n_features=2, n_redundant=0, n_informative=2)

    class Factory:
        @staticmethod
        def create(): return NClassRandomClassification()


DataSetFactory.add_factory('NClassRandomClassification', NClassRandomClassification.Factory)

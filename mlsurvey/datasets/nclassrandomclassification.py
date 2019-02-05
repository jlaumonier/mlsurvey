from sklearn.datasets import make_classification

from .dataset import DataSet
from .dataset_factory import DataSetFactory


class NClassRandomClassification(DataSet):

    def generate(self):
        self.x, self.y = make_classification()

    class Factory:
        @staticmethod
        def create(): return NClassRandomClassification()


DataSetFactory.add_factory('NClassRandomClassification', NClassRandomClassification.Factory)

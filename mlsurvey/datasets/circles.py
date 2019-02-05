from sklearn import datasets

from .dataset import DataSet
from .dataset_factory import DataSetFactory


class Circles(DataSet):

    def generate(self):
        self.x, self.y = datasets.make_circles()

    class Factory:
        @staticmethod
        def create(): return Circles()


DataSetFactory.add_factory('Circles', Circles.Factory)

from sklearn import datasets

from .dataset import DataSet
from .dataset_factory import DataSetFactory


class Moons(DataSet):

    def generate(self):
        self.x, self.y = datasets.make_moons()

    class Factory:
        @staticmethod
        def create(): return Moons()


DataSetFactory.add_factory('Moons', Moons.Factory)

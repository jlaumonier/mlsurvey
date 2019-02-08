from sklearn import datasets

from .dataset import DataSet
from .dataset_factory import DataSetFactory


class Moons(DataSet):

    def generate(self):
        n_samples = self.params.get('n_samples', 100)
        shuffle = self.params.get('shuffle', True)
        noise = self.params.get('noise', None)
        random_state = self.params.get('random_state', None)
        self.x, self.y = datasets.make_moons(n_samples=n_samples,
                                             shuffle=shuffle,
                                             noise=noise,
                                             random_state=random_state)

    class Factory:
        @staticmethod
        def create(): return Moons()


DataSetFactory.add_factory('Moons', Moons.Factory)

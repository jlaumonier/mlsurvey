from sklearn import datasets

from .dataset import DataSet
from .dataset_factory import DataSetFactory


class Circles(DataSet):

    def generate(self):
        n_samples = self.params.get('n_samples', 100)
        shuffle = self.params.get('shuffle', True)
        noise = self.params.get('noise', None)
        random_state = self.params.get('random_state', None)
        factor = self.params.get('factor', 0.8)
        self.x, self.y = datasets.make_circles(n_samples=n_samples,
                                               shuffle=shuffle,
                                               noise=noise,
                                               random_state=random_state,
                                               factor=factor)

    class Factory:
        @staticmethod
        def create(): return Circles()


DataSetFactory.add_factory('Circles', Circles.Factory)

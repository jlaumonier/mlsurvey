import numpy as np
from sklearn.datasets import make_classification

from .dataset import DataSet
from .dataset_factory import DataSetFactory


class NClassRandomClassificationWithNoise(DataSet):

    def generate(self):
        n_samples = self.params.get('n_samples', 100)
        shuffle = self.params.get('shuffle', True)
        noise = self.params.get('noise', 0)
        random_state = self.params.get('random_state', None)
        self.x, self.y = make_classification(n_features=2,
                                             n_redundant=0,
                                             n_informative=2,
                                             n_clusters_per_class=2,
                                             n_samples=n_samples,
                                             shuffle=shuffle,
                                             random_state=random_state,
                                             flip_y=noise / 10,
                                             class_sep=2 - 2 * noise
                                             )
        rng = np.random.RandomState(random_state)
        self.x += noise * 2 * rng.uniform(size=self.x.shape)

    class Factory:
        @staticmethod
        def create(t): return NClassRandomClassificationWithNoise(t)


DataSetFactory.add_factory('NClassRandomClassificationWithNoise', NClassRandomClassificationWithNoise.Factory)

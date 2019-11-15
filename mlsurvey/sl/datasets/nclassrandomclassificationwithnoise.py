import numpy as np
from sklearn.datasets import make_classification

import mlsurvey as mls
from .dataset import DataSet
from .dataset_factory import DataSetFactory


class NClassRandomClassificationWithNoise(DataSet):

    def generate(self):
        """
        Generate data of make_classification from parameters
        :return: (x, y) : data and label
        """
        n_samples = self.params.get('n_samples', 100)
        shuffle = self.params.get('shuffle', True)
        noise = self.params.get('noise', 0)
        random_state = self.params.get('random_state', None)
        x, y = make_classification(n_features=2,
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
        x += noise * 2 * rng.uniform(size=x.shape)
        data_array = np.concatenate((x, np.array([y]).T), axis=1)
        func_create_df = mls.Utils.func_create_dataframe(self.storage)
        result = func_create_df(data_array)
        return result

    class Factory:
        @staticmethod
        def create(t, storage): return NClassRandomClassificationWithNoise(t, storage)


DataSetFactory.add_factory('NClassRandomClassificationWithNoise', NClassRandomClassificationWithNoise.Factory)

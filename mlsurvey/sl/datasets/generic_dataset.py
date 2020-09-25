import numpy as np
from sklearn import datasets

import mlsurvey as mls
from .dataset import DataSet
from .dataset_factory import DataSetFactory


class GenericDataSet(DataSet):
    """
    Generic Dataset generation. Call a method, from config file, to generate a dataset.
    """

    def __init__(self, t, storage='Pandas'):
        """
        Initialize the generic dataset,
        :param t: type of the dataset, a function name into sklearn.datasets
        :param storage : the type of dataframe to store the data. By default 'Pandas' (only value of the moment)
        """
        super().__init__(t, storage)
        self.types = {'load_iris': ['load_iris', {'return_X_y': True}]}
        # default parameters if exist
        if self.t in self.types.keys():
            self.params = self.types[self.t][1].copy()

    def set_generation_parameters(self, params):
        """
        set the parameters of the generation
        :param params: dictionary containing the parameters for generation
        """
        super().set_generation_parameters(params)
        new_params = {}
        if self.t in self.types.keys():
            new_params = self.types[self.t][1].copy()
        for key, items in self.params.items():
            new_params[key] = self.params[key]
        self.params = new_params

    def generate(self):
        """
        Generate data from parameters and type.
        Raise an AttributeError if self.t is not a valid function name of sklearn.datasets
        Raise a TypeError if one parameter in self.params is not expected by the function
        :return: Dataframe with first columns are x (data) and last is y (label)
        """
        if self.t in self.types.keys():
            make_dataset = getattr(datasets, self.types[self.t][0])
        else:
            make_dataset = getattr(datasets, self.t)
        x, y = make_dataset(**self.params)
        if isinstance(x, list):
            x = np.reshape(x, (len(x), 1))
        y = np.reshape(y, (y.shape[0], 1))
        func_create_df = mls.Utils.func_create_dataframe(self.storage)
        c = np.concatenate((x, y), axis=1)
        result = func_create_df(c)
        return result

    class Factory:
        @staticmethod
        def create(t, storage): return GenericDataSet(t, storage)


DataSetFactory.add_factory('generic', GenericDataSet.Factory)

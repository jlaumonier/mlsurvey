from sklearn import datasets

from .dataset import DataSet
from .dataset_factory import DataSetFactory


class GenericDataSet(DataSet):
    """
    Generic Dataset generation. Call a method, defined in the config file, to generate a dataset.
    """

    def __init__(self, t):
        """
        Initialize the generic dataset,
        :param t: type of the dataset, a function name into sklearn.datasets
        """
        super().__init__(t)
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
        Generate data from parameters and type
        :return: (x, y) : data and label
        """
        if self.t in self.types.keys():
            make_dataset = getattr(datasets, self.types[self.t][0])
        else:
            make_dataset = getattr(datasets, self.t)
        x, y = make_dataset(**self.params)
        return x, y

    class Factory:
        @staticmethod
        def create(t): return GenericDataSet(t)


DataSetFactory.add_factory('generic', GenericDataSet.Factory)

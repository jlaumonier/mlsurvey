import os

# this line made a warning, because package is liac-arff and main file is arff :-S
import arff
import pandas as pd

import mlsurvey as mls
from .dataset import DataSet
from .dataset_factory import DataSetFactory


class FileDataSet(DataSet):

    def __init__(self, t, storage='Pandas'):
        """
        Initialize the arff file dataset,
        :param t: type of the dataset, a function name into sklearn.datasets
        :param storage : the type of dataframe to store the data. By default 'Pandas'. Other option is 'Dask'
        """
        super().__init__(t, storage)
        self.base_directory = ''

    def set_base_directory(self, d):
        """ set base directory used to search the file in generate()"""
        self.base_directory = d

    def generate(self):
        """
        load file of arff file using params ['directory'] and ['filename']. Assume that y is the last column.
        :return: (x, y) : data and label
        """
        try:
            path = self.params['directory']
            filename = self.params['filename']
        except KeyError as e:
            raise mls.exceptions.ConfigError(e)

        fullname = os.path.join(self.base_directory, path, filename)

        try:
            with open(fullname) as f:
                data = arff.load(f)
                f.close()
        except FileNotFoundError as e:
            raise mls.exceptions.ConfigError(e)

        df = pd.DataFrame(data['data'])
        # convert categorical to int
        cat_columns = df.select_dtypes(['object']).columns
        df[cat_columns] = df[cat_columns].astype('category')
        df[cat_columns] = df[cat_columns].apply(lambda c: c.cat.codes)
        return df

    class Factory:
        @staticmethod
        def create(t, storage): return FileDataSet(t, storage)


DataSetFactory.add_factory('FileDataSet', FileDataSet.Factory)

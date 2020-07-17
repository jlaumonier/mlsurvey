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
        :param storage : the type of dataframe to store the data. By default 'Pandas' (only value at the moment).
        """
        super().__init__(t, storage)
        self.base_directory = ''

    def set_base_directory(self, d):
        """ set base directory used to search the file in generate()"""
        self.base_directory = d

    def __load_arff(self, fullname):
        """ load arff file and return a pandas dataframe"""
        try:
            with open(fullname) as f:
                data = arff.load(f)
                f.close()
        except FileNotFoundError as e:
            raise mls.exceptions.ConfigError(e)

        result = mls.Utils.func_create_dataframe(self.storage)(data['data'])
        return result

    def __load_csv(self, fullname):
        """ load csv file and return a pandas dataframe"""
        try:
            result = None
            if self.storage == 'Pandas':
                result = pd.read_csv(fullname, sep='\t')
        except FileNotFoundError as e:
            raise mls.exceptions.ConfigError(e)
        return result

    def generate(self):
        """
        load file of arff file using params ['directory'] and ['filename']. Assume that y is the last column.
        :return: (x, y) : data and label
        """
        try:
            path = self.params['directory']
            filename = self.params['filename']
            (_, extension) = os.path.splitext(filename)
        except KeyError as e:
            raise mls.exceptions.ConfigError(e)

        fullname = os.path.join(self.base_directory, path, filename)

        df = None
        if extension == '.arff':
            df = self.__load_arff(fullname)
        if extension == '.csv':
            df = self.__load_csv(fullname)

        # convert categorical to int (temporary)
        cat_columns = df.select_dtypes(['object']).columns
        if self.storage == 'Pandas':
            df[cat_columns] = df[cat_columns].astype('category')
            df[cat_columns] = df[cat_columns].apply(lambda c: c.cat.codes)
        return df

    class Factory:
        @staticmethod
        def create(t, storage): return FileDataSet(t, storage)


DataSetFactory.add_factory('FileDataSet', FileDataSet.Factory)

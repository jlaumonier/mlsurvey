import os

# this line made a warning, because package is liac-arff and main file is arff :-S
import arff
import pandas as pd

import mlsurvey as mls
from .dataset import DataSet
from .dataset_factory import DataSetFactory


class FileDataSet(DataSet):

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

        fullname = os.path.join(path, filename)

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
        d = df.values
        x = d[:, :-1]
        y = d[:, -1]
        return x, y

    class Factory:
        @staticmethod
        def create(t): return FileDataSet(t)


DataSetFactory.add_factory('FileDataSet', FileDataSet.Factory)

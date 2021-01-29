import os
from typing import Dict

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

        # Columns handling
        columns = list(map(list, zip(*data['attributes'])))
        result = mls.Utils.func_create_dataframe(self.storage)(data['data'], columns=columns[0])
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

    def __load_json(self, fullname, func_params: dict):
        """ load json file and return a pandas dataframe"""
        try:
            result = None

            if self.storage == 'Pandas':
                result = pd.read_json(fullname, **func_params)
        except FileNotFoundError as e:
            raise mls.exceptions.ConfigError(e)
        return result

    def __load_xlsx(self, fullname):
        """
        load xlsx file and return a pandas dataframe. merge all sheets if the file contains multiple sheets
        :param fullname
        """

        def _merge_if_multiples_sheets(data):
            # merge all the sheets data into one DataFrame
            if isinstance(data, Dict):
                list_sheets = data.keys()
                # concat adding a index as list of sheets
                data = pd.concat(data, keys=list_sheets)
                # put the indexed into the dataframe
                data.reset_index(inplace=True)
                # drop the level 1 index
                data.drop('level_1', axis=1, inplace=True)
                # rename the level 0 index as Sheet
                data.rename(columns={'level_0': 'Sheet'}, inplace=True)
                # move 'Sheet' column at the end
                list_cols = data.columns.tolist()
                list_cols.insert(len(list_cols), list_cols.pop(list_cols.index('Sheet')))
                data = data.reindex(list_cols, axis=1)
            return data

        try:
            result = None
            if self.storage == 'Pandas':
                result = pd.read_excel(fullname, sheet_name=None, engine='openpyxl')
                result = _merge_if_multiples_sheets(result)
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

        func_params = {}
        if 'func_params' in self.params:
            func_params = self.params['func_params']

        fullname = os.path.join(self.base_directory, path, filename)

        df = None
        if extension == '.arff':
            df = self.__load_arff(fullname)
        if extension == '.csv':
            df = self.__load_csv(fullname)
        if extension == '.xlsx':
            df = self.__load_xlsx(fullname)
        if extension == '.json':
            df = self.__load_json(fullname, func_params)

        return df

    class Factory:
        @staticmethod
        def create(t, storage): return FileDataSet(t, storage)


DataSetFactory.add_factory('FileDataSet', FileDataSet.Factory)

import json
import os

import pandas as pd

import mlsurvey as mls


class FileOperation:

    @classmethod
    def save_dict_as_json(cls, filename, directory, data):
        """
        Save a dictionary into a json file. Transform the dictionary into mlsurvey json format (for lists).
        Create the directory of not exists
        :param filename: name of the file
        :param directory: directory to save the file
        :param data: data to save into the file
        """
        os.makedirs(directory, exist_ok=True)
        full_path = os.path.join(directory, filename)
        with open(full_path, 'w') as outfile:
            json.dump(mls.Utils.transform_to_json(data), outfile)
        outfile.close()

    @classmethod
    def load_json_as_dict(cls, filename, directory, tuple_to_string=False):
        """
        Load a json file (mlsurvey json format) into dictionary
        :param filename: name of the file
        :param directory: directory to load the file
        :param tuple_to_string: if True the tuple identified with "__type__": "__tuple__" are store as string in the
               dictionary. If False, the tuple is converted to a tuple type
        :return: the dictionary
        """
        full_path = os.path.join(directory, filename)
        with open(full_path, 'r') as infile:
            data = mls.Utils.transform_to_dict(json.load(infile), tuple_to_string)
        return data

    @classmethod
    def save_hdf(cls, filename, directory, data, params=None):
        """
        Save a dataframe into a hdf file.
        Create the directory of not exists
        :param filename: name of the file
        :param directory: directory to save the file
        :param data: data to save into the file
        :param params: parameters passed to the to_json() method
        """
        if params is None:
            params = {}
        os.makedirs(directory, exist_ok=True)
        full_path = os.path.join(directory, filename)
        data.to_hdf(full_path, 'key', mode='w', **params)

    @classmethod
    def read_hdf(cls, filename, directory, df_format):
        """
        Load a dataframe from a hdf file.
        :param filename: name of the file
        :param directory: directory to load the file
        :param df_format: 'Pandas'
        """
        full_path = os.path.join(directory, filename)
        data = None
        if df_format == 'Pandas':
            data = pd.read_hdf(full_path, 'key')
        return data

    @classmethod
    def save_json(cls, filename, directory, data, params=None):
        """
        Save a dataframe into a json file.
        Create the directory of not exists
        :param filename: name of the file
        :param directory: directory to save the file
        :param data: data to save into the file
        :param params: parameters passed to the to_json() method
        """
        if params is None:
            params = {}
        os.makedirs(directory, exist_ok=True)
        full_path = os.path.join(directory, filename)
        with open(full_path, 'w', encoding='utf-8') as file:
            data.to_json(file, force_ascii=False, **params)

    @classmethod
    def read_json(cls, filename, directory, df_format):
        """
        Load a dataframe from a json file.
        :param filename: name of the file
        :param directory: directory to load the file
        :param df_format: 'Pandas'
        """
        full_path = os.path.join(directory, filename)
        data = None
        if df_format == 'Pandas':
            data = pd.read_json(full_path)
        return data




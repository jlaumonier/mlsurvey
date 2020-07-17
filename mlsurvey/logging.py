import datetime
import os
import random

import joblib
import pandas as pd

import mlsurvey as mls


class Logging:

    def __init__(self, dir_name=None, base_dir='logs/'):
        """
        initialize machine learning logging by creating dir_name/ in base_dir/
        :param dir_name: name of the log directory for this instance
        :param base_dir: name of the base directory for all logging
        """
        self.base_dir = base_dir
        # adding a random number to avoid the creating at the same microsecond !!
        salt_random_number = random.randint(0, 9)
        if dir_name is None:
            dir_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f") + '-' + str(salt_random_number)
        self.dir_name = dir_name
        self.directory = os.path.join(self.base_dir, dir_name, '')

    def save_input(self, inpts):
        """
        save the dictionary of inputs into a file
        :param inpts: dictionary of inputs
        """
        output = {}
        for k, v in inpts.items():
            if v is not None:
                filename = k + '.h5'
                mls.FileOperation.save_hdf(filename, self.directory, v.df)
                df_format = ''
                if isinstance(v.df, pd.DataFrame):
                    df_format = 'Pandas'
                output[k] = {'data_path': filename,
                             'df_format': df_format,
                             'metadata': v.to_dict()
                             }
            else:
                output[k] = None
        self.save_dict_as_json('input.json', output)

    def load_input(self, filename):
        """
        load inputs from json file. The file may contains multiple input {"input1": input1, "input2": input2"}
        :param filename: the name of the file
        :return: dictionary containing each inputs
        """
        data = self.load_json_as_dict(filename)
        result = {}
        for k, v in data.items():
            df = mls.FileOperation.read_hdf(v['data_path'], self.directory, v['df_format'])
            i = mls.sl.models.DataFactory.create_data_from_dict(v['df_format'], v['metadata'], df)
            result[k] = i
        return result

    def save_dict_as_json(self, filename, d):
        """ save a dictionary into a json file"""
        mls.FileOperation.save_dict_as_json(filename, self.directory, d)

    def load_json_as_dict(self, filename):
        """ load a dictionary from a json file"""
        data = mls.FileOperation.load_json_as_dict(filename, self.directory)
        return data

    def save_classifier(self, classifier):
        """ save a scikitlearn classifier"""
        os.makedirs(self.directory, exist_ok=True)
        joblib.dump(classifier, self.directory + 'model.joblib')

    def load_classifier(self):
        """ load a scikitlearn classifier"""
        os.makedirs(self.directory, exist_ok=True)
        return joblib.load(self.directory + 'model.joblib')

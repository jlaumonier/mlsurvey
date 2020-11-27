import datetime
import os
import random

import joblib
import pandas as pd
import mlflow

import mlsurvey as mls


class Logging:

    def __init__(self, dir_name=None, base_dir='logs/', mlflow_run_id=None):
        """
        initialize machine learning logging by creating dir_name/ in base_dir/
        :param dir_name: name of the log directory for this instance
        :param base_dir: name of the base directory for all logging
        :param mlflow_run_id: mlflow run id of the corresponding run
        """
        self.base_dir = base_dir
        # adding a random number to avoid the creating at the same microsecond !!
        salt_random_number = random.randint(0, 9)
        if dir_name is None:
            dir_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f") + '-' + str(salt_random_number)
        self.dir_name = dir_name
        self.directory = os.path.join(self.base_dir, dir_name)
        self.mlflow_client = mlflow.tracking.MlflowClient()
        self.mlflow_run_id = mlflow_run_id

    def save_input(self, inpts, metadata_filename='input.json'):
        """
        save the dictionary of inputs into a file (.json for metadata, h5 for data). One json may contain multiple data
        :param inpts: dictionary of inputs
        :param metadata_filename: name of the json file containing metadata
        """
        output = {}
        for k, v in inpts.items():
            if v is not None:
                filename = k + '-content'
                h5_filename = filename + '.h5'
                json_filename = filename + '.json'
                mls.FileOperation.save_hdf(h5_filename, self.directory, v.df)
                mls.FileOperation.save_json(json_filename, self.directory, v.df)
                if self.mlflow_run_id is not None:
                    self.mlflow_client.log_artifact(self.mlflow_run_id, os.path.join(self.directory, json_filename))
                df_format = ''
                if isinstance(v.df, pd.DataFrame):
                    df_format = 'Pandas'
                output[k] = {'data_path': h5_filename,
                             'df_format': df_format,
                             'metadata': v.to_dict(),
                             'shape': v.df.shape
                             }
            else:
                output[k] = None
        self.save_dict_as_json(metadata_filename, output)

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
        if self.mlflow_run_id is not None:
            self.mlflow_client.log_artifact(self.mlflow_run_id, os.path.join(self.directory, filename))

    def load_json_as_dict(self, filename):
        """ load a dictionary from a json file"""
        data = mls.FileOperation.load_json_as_dict(filename, self.directory)
        return data

    def save_classifier(self, classifier, filename='model.joblib'):
        """ save a scikitlearn classifier"""
        os.makedirs(self.directory, exist_ok=True)
        joblib.dump(classifier, os.path.join(self.directory, filename))
        if self.mlflow_run_id is not None:
            self.mlflow_client.log_artifact(self.mlflow_run_id, os.path.join(self.directory, filename))

    def load_classifier(self, filename='model.joblib'):
        """ load a scikitlearn classifier"""
        os.makedirs(self.directory, exist_ok=True)
        return joblib.load(os.path.join(self.directory, filename))

    def save_plotly_figures(self, dict_figures, sub_directory):
        """ save a list of plotly figure into image files into sub_directory"""
        target_dir = os.path.join(self.directory, sub_directory)
        for (filename, figure) in dict_figures.items():
            mls.FileOperation.save_plotly_figure(filename, target_dir, figure)
            if self.mlflow_run_id is not None:
                self.mlflow_client.log_artifact(self.mlflow_run_id, os.path.join(target_dir, filename))

    def log_config(self, filename, config_dict):
        """ log config into file and mlflow"""
        self.save_dict_as_json(filename, config_dict)
        # log config into mlflow
        if self.mlflow_run_id is not None:
            params = mls.Utils.flatten_dict(config_dict, separator='.')
            for key, value in params.items():
                if len(key) >= 28:
                    key = key[28:]
                self.mlflow_client.log_param(self.mlflow_run_id, key, value)

    def log_metrics(self, filename, metric_dict):
        """ log metric into file and mlflow"""
        self.save_dict_as_json(filename, metric_dict)
        # log metrics into mlflow
        if self.mlflow_run_id is not None:
            metrics = mls.Utils.flatten_dict(metric_dict, separator='.')
            for key, value in metrics.items():
                if isinstance(value, int) or isinstance(value, float):
                    self.mlflow_client.log_metric(self.mlflow_run_id, key, value)

    @staticmethod
    def msg(msg: str, level):
        """ Log a msg in the python classical log"""
        print(level, " ", msg)


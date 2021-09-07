import datetime
import os
import random
import shutil
import pathlib

import joblib
import pandas as pd
import mlflow

import mlsurvey as mls


class Logging:

    def __init__(self, dir_name=None, base_dir='logs/',
                 mlflow_log=False, mlflow_tracking_uri=None, mlflow_xp_name=None):
        """
        initialize machine learning logging by creating dir_name/ in base_dir/
        :param dir_name name of the log directory for this instance
        :param base_dir name of the base directory for all logging
        :param mlflow_log is the log will be recorded to mlflow
        :param mlflow_tracking_uri tracking uri to log to an mlflow server (Not unit tested)
        :param mlflow_xp_name name of the experiement. log dir_name by default (if None)
        """
        self.base_dir = base_dir
        # adding a random number to avoid the creating at the same microsecond !!
        salt_random_number = random.randint(0, 9)
        if dir_name is None:
            dir_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f") + '-' + str(salt_random_number)
        self.dir_name = dir_name
        self.sub_dir = ''
        # mlflow initialization
        self.is_log_to_mlflow = mlflow_log
        self.mlflow_client = None
        self.mlflow_experiment = None
        self.mlflow_current_run = None
        self.mlflow_runs = None
        if self.is_log_to_mlflow:
            self.mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)
            if mlflow_xp_name is None:
                mlflow_xp_name = self.dir_name
            self.mlflow_experiment = self.mlflow_client.get_experiment_by_name(mlflow_xp_name)
            if not self.mlflow_experiment:
                xp_id = self.mlflow_client.create_experiment(mlflow_xp_name)
                self.mlflow_experiment = self.mlflow_client.get_experiment(xp_id)
            self.mlflow_runs = []
            # Warning : during the execution, this instance mlflow_run does not contains
            # other data than those at the creation. use mlflow_client.get_run(mlflow_run.info.run_id) to have
            # the actual run with data
            self.new_mlflow_run()

    def set_sub_dir(self, sub_dir):
        self.sub_dir = sub_dir

    @property
    def directory(self):
        return os.path.join(self.base_dir, self.dir_name, self.sub_dir)

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
                if self.is_log_to_mlflow:
                    self.mlflow_client.log_artifact(self.mlflow_current_run.info.run_id,
                                                    os.path.join(self.directory, json_filename),
                                                    self.sub_dir)
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
        result = mls.FileOperation.load_input(filename=filename, directory=self.directory)
        return result

    def save_dict_as_json(self, filename, d):
        """ save a dictionary into a json file"""
        mls.FileOperation.save_dict_as_json(filename, self.directory, d)
        if self.is_log_to_mlflow:
            self.mlflow_client.log_artifact(self.mlflow_current_run.info.run_id,
                                            os.path.join(self.directory, filename),
                                            self.sub_dir)

    def load_json_as_dict(self, filename):
        """ load a dictionary from a json file"""
        data = mls.FileOperation.load_json_as_dict(filename, self.directory)
        return data

    def save_classifier(self, classifier, filename='model.joblib'):
        """ save a scikitlearn classifier"""
        os.makedirs(self.directory, exist_ok=True)
        joblib.dump(classifier, os.path.join(self.directory, filename))
        if self.is_log_to_mlflow:
            self.mlflow_client.log_artifact(self.mlflow_current_run.info.run_id,
                                            os.path.join(self.directory, filename),
                                            self.sub_dir)

    def load_classifier(self, filename='model.joblib', name_is_full_path=False):
        """ load a scikitlearn classifier"""
        os.makedirs(self.directory, exist_ok=True)
        directory = '' if name_is_full_path else self.directory
        return joblib.load(os.path.join(directory, filename))

    def save_plotly_figures(self, dict_figures, plot_directory):
        """ save a list of plotly figure into image files into sub_directory"""
        target_dir = os.path.join(self.directory, plot_directory)
        for (filename, figure) in dict_figures.items():
            mls.FileOperation.save_plotly_figure(filename, target_dir, figure)
            if self.is_log_to_mlflow:
                self.mlflow_client.log_artifact(self.mlflow_current_run.info.run_id,
                                                os.path.join(target_dir, filename),
                                                os.path.join(self.sub_dir, plot_directory))

    def log_config(self, filename, config_dict):
        """ log config into file and mlflow"""
        self.save_dict_as_json(filename, config_dict)
        # log config into mlflow
        if self.is_log_to_mlflow:
            params = mls.Utils.flatten_dict(config_dict, separator='.')
            for key, value in params.items():
                if len(key) >= 28:
                    key = key[28:]
                self.mlflow_client.log_param(self.mlflow_current_run.info.run_id, key, value)

    def log_metrics(self, filename, metric_dict):
        """ log metric into file and mlflow"""
        self.save_dict_as_json(filename, metric_dict)
        # log metrics into mlflow
        if self.is_log_to_mlflow:
            metrics = mls.Utils.flatten_dict(metric_dict, separator='.')
            for key, value in metrics.items():
                if isinstance(value, int) or isinstance(value, float):
                    self.mlflow_client.log_metric(self.mlflow_current_run.info.run_id, key, value)

    def copy_source_tree(self, source, dest_dir):
        """ copy a directory (recursively) or file into self.directory/'sources/' """
        if os.path.isdir(source):
            shutil.copytree(source, os.path.join(self.directory, 'sources', dest_dir, pathlib.PurePath(source).name))
        if os.path.isfile(source):
            dest_full_dir = os.path.join(self.directory, 'sources', dest_dir)
            os.makedirs(dest_full_dir, exist_ok=True)
            shutil.copyfile(source, os.path.join(dest_full_dir, pathlib.PurePath(source).name))

    def terminate_mlflow(self):
        """terminate mlflow run"""
        # Not tested
        url = ''
        if self.is_log_to_mlflow:
            self.mlflow_client.set_terminated(self.mlflow_current_run.info.run_id)
            url = "experiments/" + str(self.mlflow_experiment.experiment_id) + "/runs/" + \
                  str(self.mlflow_current_run.info.run_id)
        return url

    def new_mlflow_run(self):
        """ create a new mlflow run in the current experiement"""
        self.mlflow_current_run = self.mlflow_client.create_run(self.mlflow_experiment.experiment_id)
        self.mlflow_runs.append(self.mlflow_current_run)

    def save_mlflow_artifact(self, path_to_file):
        """ save an existing file from log directory (or subdirectory) into mlflow artifacts (current run)"""
        if self.is_log_to_mlflow:
            self.mlflow_client.log_artifact(self.mlflow_current_run.info.run_id,
                                            os.path.join(self.directory, path_to_file),
                                            self.sub_dir)

    @staticmethod
    def msg(msg: str, level):
        """ Log a msg in the python classical log"""
        print(level, " ", msg)

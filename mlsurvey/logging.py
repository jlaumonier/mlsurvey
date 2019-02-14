import datetime
import json
import os

import joblib

import mlsurvey as mls


class Logging:

    def __init__(self, dir_name=None, base_dir='logs/'):
        """
        initialize machine learning logging by creating dir_name/ in base_dir/
        :param dir_name: name of the log directory for this instance
        :param base_dir: name of the base directory for all loggingl
        """
        self.base_dir = base_dir
        if dir_name is None:
            dir_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
        self.dir_name = dir_name
        self.directory = self.base_dir + dir_name + '/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def save_input(self, inpts):
        """
        save the dictionary of inputs into a file
        :param inpts: dictionary of inputs
        """
        output = {}
        for k, v in inpts.items():
            output[k] = v.to_dict()
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
            i = mls.Input()
            i.from_dict(v)
            result[k] = i
        return result

    def save_dict_as_json(self, filename, d):
        """ save a dictionary into a json file"""
        with open(self.directory + filename, 'w') as outfile:
            json.dump(d, outfile)
        outfile.close()

    def load_json_as_dict(self, filename):
        """ load a dictionary from a json file"""
        with open(self.directory + filename, 'r') as infile:
            data = json.load(infile)
        return data

    def save_classifier(self, classifier):
        """ save a scikitlearn classifier"""
        joblib.dump(classifier, self.directory + 'model.joblib')

    def load_classifier(self):
        """ load a scikitlearn classifier"""
        return joblib.load(self.directory + 'model.joblib')

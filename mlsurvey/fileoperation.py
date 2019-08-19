import json
import os

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
    def load_json_as_dict(cls, filename, directory):
        """
        Load a json file (mlsurvey json format) into dictionary
        :param filename: name of the file
        :param directory: directory to load the file
        :return: the dictionary
        """
        full_path = os.path.join(directory, filename)
        with open(full_path, 'r') as infile:
            data = mls.Utils.transform_to_dict(json.load(infile))
        return data

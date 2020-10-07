import ast
import hashlib
import importlib

import numpy as np
import pandas as pd


class Utils:
    """
    Utils functions
    """

    @classmethod
    def md5_file(cls, filename):
        """
        Calculate the md5 of a file
        thanks Andres Torres https://www.pythoncentral.io/hashing-files-with-python/
        Raise FileNotFoundError if the file does not exist
        """
        blocksize = 65536
        hasher = hashlib.md5()
        with open(filename, 'rb') as afile:
            buf = afile.read(blocksize)
            while len(buf) > 0:
                hasher.update(buf)
                buf = afile.read(blocksize)
        return hasher.hexdigest()

    @classmethod
    def import_from_dotted_path(cls, dotted_names):
        """ import_from_dotted_path('foo.bar') -> from foo import bar; return bar
        """
        module_name, class_name = dotted_names.rsplit('.', 1)
        module = importlib.import_module(module_name)
        handler_class = getattr(module, class_name)
        return handler_class

    @classmethod
    def make_meshgrid(cls, x, y, h=.02):
        """
        Create a mesh of points to plot in
        (src, thanks : https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html)
        :param x: data to base x-axis meshgrid on (type numpy.ndarray)
        :param y: data to base y-axis meshgrid on (type numpy.ndarray)
        :param h: stepsize for meshgrid, optional
        :return: xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

    @classmethod
    def transform_to_dict(cls, dictionary: dict, tuple_to_string=False):
        """
        Transform a dictionary containing dictionary such as
        { "__type__": "__tuple__", "__value__": "(1, 2, 3)"}
        to dictionary containing the real type (tuple)
        :param dictionary: dictionary containing __tuple__ values
        :param tuple_to_string: if True the tuple identified with "__type__": "__tuple__" are store as string in the
            dictionary. If False, the tuple is converted to a tuple type
        :return dictionary containing the real type
        """
        def change_one_dict_element(value):
            if '__type__' in value:
                if value['__type__'] == '__tuple__':
                    if tuple_to_string:
                        result_element = value['__value__']
                    else:
                        result_element = ast.literal_eval(value['__value__'])
                        if not isinstance(result_element, tuple):
                            raise TypeError(v['__value__'] + " is not a tuple")
                else:
                    result_element = Utils.transform_to_dict(value, tuple_to_string)
            else:
                result_element = Utils.transform_to_dict(value, tuple_to_string)
            return result_element

        result = dictionary.copy()
        for k, v in result.items():
            if isinstance(v, dict):
                result[k] = change_one_dict_element(v)
            if isinstance(v, list):
                result[k] = []
                for e in v:
                    if isinstance(e, dict):
                        result[k].append(change_one_dict_element(e))
                    else:
                        result[k].append(e)
        return result

    @classmethod
    def transform_to_json(cls, dictionary):
        """
        Transform a dictionary containing tuple to dictionary
        such as { "__type__": "__tuple__", "__value__": "(1, 2, 3)"}
        :param dictionary: dictionary containing tuple
        :return dictionary containing __tuple__ values
        """
        def change_one_dict_element(value):
            result_element = {'__type__': '__tuple__', '__value__': value.__str__()}
            return result_element

        result = dictionary.copy()
        for k, v in result.items():
            if isinstance(v, tuple):
                result[k] = change_one_dict_element(v)
            if isinstance(v, dict):
                result[k] = Utils.transform_to_json(v)
            if isinstance(v, list):
                result[k] = []
                for e in v:
                    if isinstance(e, tuple):
                        result[k].append(change_one_dict_element(e))
                    else:
                        if isinstance(e, dict):
                            result[k].append(Utils.transform_to_json(e))
                        else:
                            result[k].append(e)
        return result

    @classmethod
    def check_dict_python_ready(cls, dictionary):
        """
        Check if a dictionary (and nested) does not contains a __type__ key,
        which means is not ready to be handle by python
        :param dictionary: the dictionary to check
        :return: False if the dictionary contains one __type__ key, True otherwise
        """
        result = True
        for k, v in dictionary.items():
            if not isinstance(v, list):
                v = [v]
            for e in v:
                if isinstance(e, dict):
                    if '__type__' in e:
                        result = False
                    else:
                        result = result & Utils.check_dict_python_ready(e)
        return result

    @classmethod
    def flatten_dict(cls, dictionary, separator='_', prefix=''):
        """SRC : https://www.geeksforgeeks.org/python-convert-nested-dictionary-into-flattened-dictionary/"""
        result = {prefix + separator + k if prefix else k: v
                  for kk, vv in dictionary.items()
                  for k, v in Utils.flatten_dict(vv, separator, kk).items()
                  } if isinstance(dictionary, dict) else {prefix: dictionary}
        return result

    @classmethod
    def func_create_dataframe(cls, storage):
        """ return the function that create a DataFrame from an array"""
        if storage == 'Pandas':
            return pd.DataFrame

    @classmethod
    def is_dataframe_empty(cls, df):
        result = True
        if isinstance(df, pd.DataFrame):
            result = df.empty
        return result

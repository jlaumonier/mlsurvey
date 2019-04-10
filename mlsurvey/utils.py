import hashlib
import importlib
import itertools

import numpy as np


class Utils:

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
    def dict_generator_cartesian_product(cls, source):
        """ get a dictionary containing lists and calculate the cartesian product of these lists.
            return a generator of dictionaries
        """
        keys = []
        vals = []
        for k, v in source.items():
            keys.append(k)
            if isinstance(v, list):
                vals.append(v)
            else:
                vals.append([v])
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

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
        """Create a mesh of points to plot in
        (src, thanks : https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html)

        Parameters
        ----------
        x: data to base x-axis meshgrid on (type numpy.ndarray)
        y: data to base y-axis meshgrid on (type numpy.ndarray)
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

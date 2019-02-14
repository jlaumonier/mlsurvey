import hashlib
import itertools


class Utils:

    @classmethod
    def md5_file(cls, filename):
        """
        Calculate the md5 of a file
        thanks Andres Torres https://www.pythoncentral.io/hashing-files-with-python/
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

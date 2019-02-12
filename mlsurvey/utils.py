import hashlib


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

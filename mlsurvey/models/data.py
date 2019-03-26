import numpy as np


class Data:

    def __init__(self):
        self.x = np.asarray([])
        self.y = np.asarray([])

    def to_dict(self):
        """
        Transform the input to a dictionary containing 'data.x' and 'data.y'
        :return: a dictionary
        """
        result = {'data.x': self.x.tolist(), 'data.y': self.y.tolist()}
        return result

    @staticmethod
    def from_dict(d):
        """
        Transform a dictionary containing 'data.x' and 'data.y' to the input object
        :param d: source dictionary
        """
        result = Data()
        result.x = np.array(d['data.x'])
        result.y = np.array(d['data.y'])
        return result

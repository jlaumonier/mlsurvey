import numpy as np


class Input:

    def __init__(self):
        self.x = np.asarray([])
        self.y = np.asarray([])

    def set_data(self, dataset):
        self.x = dataset.x
        self.y = dataset.y

    def to_dict(self):
        """
        Transform the input to a dictionary containing 'input.x' and 'input.y'
        :return: a dictionary
        """
        result = {'input.x': self.x.tolist(), 'input.y': self.y.tolist()}
        return result

    def from_dict(self, d):
        """
        Transform a dictionary containing 'input.x' and 'input.y' to the input object
        :param d: source dictionary
        """
        self.x = np.array(d['input.x'])
        self.y = np.array(d['input.y'])

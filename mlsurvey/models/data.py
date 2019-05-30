import numpy as np

import mlsurvey as mls


class Data:

    def __init__(self):
        self.x = np.asarray([])
        self.y = np.asarray([])
        self.y_pred = np.asarray([])

    def to_dict(self):
        """
        Transform the input to a dictionary containing 'data.x' and 'data.y'
        :return: a dictionary
        """
        result = {'data.x': self.x.tolist(), 'data.y': self.y.tolist(), 'data.y_pred': self.y_pred.tolist()}
        return result

    @staticmethod
    def from_dict(d):
        """
        Transform a dictionary containing 'data.x' and 'data.y' to the input object
        :param d: source dictionary
        Raise mlsurvey.exception.ModelError if data.x or data.y is not present
        """
        result = Data()
        try:
            result.x = np.array(d['data.x'])
            result.y = np.array(d['data.y'])
            result.y_pred = np.array(d['data.y_pred'])
        except KeyError as e:
            raise mls.exceptions.ModelError(e)
        return result

    def merge_all(self):
        result = np.concatenate((self.x,
                                 self.y.reshape((-1, 1)),
                                 self.y_pred.reshape((-1, 1))),
                                axis=1)
        return result

    def copy(self):
        """
        copy the object into another
        :return: the new object
        """
        result = Data()
        result.x = self.x.copy()
        result.y = self.y.copy()
        result.y_pred = self.y_pred.copy()
        return result

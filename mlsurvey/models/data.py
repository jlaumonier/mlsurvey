import numpy as np

import mlsurvey as mls


class Data:

    def __init__(self):
        self.__x = np.asarray([])
        self.__y = np.asarray([])
        self.__y_pred = np.asarray([])

    def set_data(self, x_values, y_values):
        """
        set the x and y data with values
        """
        if x_values is not None:
            self.__x = x_values
        if y_values is not None:
            self.__y = y_values

    def set_pred_data(self, y_pred_values):
        """
        set the y predicted data with values
        """
        self.__y_pred = y_pred_values

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def y_pred(self):
        return self.__y_pred

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
            result.set_data(np.array(d['data.x']), np.array(d['data.y']))
            result.set_pred_data(np.array(d['data.y_pred']))
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
        result.set_data(self.x.copy(), self.y.copy())
        result.set_pred_data(self.y_pred.copy())
        return result

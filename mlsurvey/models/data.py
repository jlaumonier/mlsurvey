import numpy as np
import pandas as pd

import mlsurvey as mls


class Data:

    def __init__(self, x, y=None, y_pred=None, columns=None):
        if columns is None and x.ndim > 1:
            columns = []
            for idx_col in range(x.shape[1]):
                col_name = 'C' + str(idx_col)
                columns.append(col_name)
        self.__inner_data = pd.DataFrame(data=x, columns=columns)
        self.set_data(None, y)
        self.set_pred_data(y_pred)

    def set_data(self, x_values, y_values):
        """
        set the x and y data with values. size of the x and y value must be the same size as existing data
        """
        if x_values is not None:
            self.__inner_data.iloc[:, 0:-2] = x_values
        if y_values is not None:
            self.__inner_data['target'] = y_values

    def set_pred_data(self, y_pred_values):
        """
        set the y predicted data with values
        """
        self.__inner_data['target_pred'] = pd.Series(y_pred_values)

    def add_column_in_data(self, new_column, column_name=None):
        """
        add a new column at the end of the data
        """
        if column_name is None:
            column_name = 'C' + str(len(self.__inner_data.columns) - 2)
        self.__inner_data.insert(len(self.__inner_data.columns) - 2, column_name, new_column)

    @property
    def x(self):
        return self.__inner_data.iloc[:, 0:-2].to_numpy()

    @property
    def y(self):
        return self.__inner_data['target'].to_numpy()

    @property
    def y_pred(self):
        return self.__inner_data['target_pred'].to_numpy()

    @property
    def df(self):
        return self.__inner_data

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
        try:
            result = Data(x=np.array(d['data.x']),
                          y=np.array(d['data.y']),
                          y_pred=np.array(d['data.y_pred']))
        except KeyError as e:
            raise mls.exceptions.ModelError(e)
        return result

    def merge_all(self):
        result = self.__inner_data.values
        return result

    def copy(self):
        """
        copy the object into another
        :return: the new object
        """
        result = Data(x=self.x.copy(),
                      y=self.y.copy(),
                      y_pred=self.y_pred.copy())
        return result

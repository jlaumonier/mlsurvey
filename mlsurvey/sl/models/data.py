from abc import ABC, abstractmethod

import mlsurvey as mls


class Data(ABC):

    def __init__(self, df, df_contains='xy', y_col_name=None, y_pred_col_name=None):
        """
        :param df: dataframe containing data
        :param df_contains: 'xy' or 'xyypred' if the dataframe contains x, y or x, y and y_pred
        :param y_col_name: name of the y column.
                           By default 'target' and the last column in the dataframe  (if df_contains = 'xy')
        :param y_pred_col_name: name the y_pred column.
                                if df_contains='xyypred', 'target_pred' and the last column in the dataframe
        """
        if y_col_name is None:
            y_col_name = 'target'
        if y_pred_col_name is None:
            y_pred_col_name = 'target_pred'

        self.df_contains = df_contains
        if df_contains == 'xy':
            self.max_x_column = -1
        if df_contains == 'xyypred':
            self.max_x_column = -2

        empty = mls.Utils.is_dataframe_empty(df)
        if not empty:
            if not isinstance(df.columns[0], str):
                # if columns are not set in the dataframe, we create some names
                columns = []
                for idx_col in range(df.shape[1] + self.max_x_column):
                    col_name = 'C' + str(idx_col)
                    columns.append(col_name)
                columns.append(y_col_name)
                if df_contains == 'xyypred':
                    columns.append(y_pred_col_name)
                df.columns = columns
            self._inner_data = df
            self.y_col_name = y_col_name
            self.y_pred_col_name = y_pred_col_name
        else:
            raise mls.exceptions.ModelError('Empty dataframe is not allowed')

    def set_pred_data(self, y_pred_values, y_pred_col_name=None):
        """
        set the y predicted data with values
        :param y_pred_values: Pandas dataframe containing one column, the y predicted by a classifier.
                              If __inner_data is a pandas Dataframe, y_pred_value must be also a pandas dataframe
        :param y_pred_col_name: name of the column into the final dataset
        """
        if y_pred_col_name is not None:
            # if col name is given => use it as the new name of the column
            self.y_pred_col_name = y_pred_col_name
            y_pred_values.columns = [self.y_pred_col_name]
        else:
            if not isinstance(y_pred_values.columns[0], str):
                # if col name is not given and set into the dataframe => use the default defined in __init__()
                y_pred_values.columns = [self.y_pred_col_name]
            else:
                # col name is not given but is set into the dataframe => use the dataframe column name
                self.y_pred_col_name = y_pred_values.columns[0]
        self._inner_data = self._inner_data.merge(y_pred_values, left_index=True, right_index=True)
        self.df_contains = 'xyypred'
        self.max_x_column = -2

    def add_calculated_column(self, condition, on_column, new_column_name):
        """
        Adding a column at the end of the dataframe containing a calculated value with condition
        :param condition: string that represent a condition apply by lambda x: (ex : 'x > 3')
        :param on_column: string containing the name of the column to apply the condition
        :param new_column_name: string containing the name of the new column containing the values
        """
        columns = self._inner_data.columns
        self._inner_data[new_column_name] = self._inner_data[on_column].map(eval('lambda x:' + condition))
        new_columns = columns.insert(self.max_x_column, new_column_name)
        self._inner_data = self._inner_data[new_columns]

    @property
    @abstractmethod
    def x(self):
        ...

    @property
    @abstractmethod
    def x_df(self):
        ...

    @property
    @abstractmethod
    def y(self):
        ...

    @property
    @abstractmethod
    def y_pred(self):
        ...

    @property
    def df(self):
        return self._inner_data

    def to_dict(self):
        """
        Transform the input to a dictionary containing all data informations EXCEPT data itself
        :return: a dictionary
        """
        result = {'df_contains': self.df_contains,
                  'y_col_name': self.y_col_name,
                  'y_pred_col_name': self.y_pred_col_name}
        return result

    @staticmethod
    @abstractmethod
    def from_dict(dictionary, df):
        """
        Transform a dictionary to the input object
        :param dictionary: source dictionary
        :param df: dataframe containing data
        Raise mlsurvey.exception.ModelError if dictionary does not contain all keys
        """
        ...

    def merge_all(self):
        """
        transform the dataframe to an array containing all values
        Only for a pandas dataframe.
        :return: numpy.array containing all data
        """
        result = self._inner_data.values
        return result

    @abstractmethod
    def copy_with_new_data(self, data_array):
        """
        create a new instance of data with the same parameters but with different data values
        :param data_array: new data values as numpy array
        :return: an instance of data
        """
        ...

    @abstractmethod
    def copy_with_new_data_dataframe(self, dataframe):
        """
        create a new instance of data with the same parameters but with different data values
        :param dataframe: new data values as dataframe
        :return: an instance of data
        """
        ...

import dask.dataframe as dd
import numpy as np
import pandas as pd

import mlsurvey as mls


class Data:

    # def __init__(self, x, y=None, y_pred=None, columns=None):
    #     if columns is None and x.ndim > 1:
    #         columns = []
    #         for idx_col in range(x.shape[1]):
    #             col_name = 'C' + str(idx_col)
    #             columns.append(col_name)
    #     self.__inner_data = pd.DataFrame(data=x, columns=columns)
    #     print(self.x.shape)
    #     self.set_data(None, y)
    #     print(self.x.shape)
    #     self.set_pred_data(y_pred)
    #     print(self.x.shape)

    def __init__(self, df, df_contains='xy', y_col_name=None, y_pred_col_name=None):
        """
        :param df: dataframe containing data
        :param df_contains: 'xy' or 'xyypred' if the dataframe contains x, y or x, y and y_pred
        :param y_col_name:
        :param y_pred_col_name:
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

        empty = False
        if isinstance(df, pd.DataFrame):
            empty = df.empty
        if isinstance(df, dd.DataFrame) or isinstance(df, dd.Series):
            empty = (df.npartitions == 0)
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
            self.__inner_data = df
            self.y_col_name = y_col_name
            self.y_pred_col_name = y_pred_col_name
        else:
            raise mls.exceptions.ModelError('Empty dataframe is not allowed')

    def set_pred_data(self, y_pred_values, y_pred_col_name=None):
        """
        set the y predicted data with values
        :param y_pred_values: Pandas or Dask dataframe containing one column, the y predicted by a classifier.
                              Must not be a Dask Serie
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
        if isinstance(self.__inner_data, pd.DataFrame) and isinstance(y_pred_values, dd.DataFrame):
            raise mls.exceptions.ModelError("__inner_data and y_pred_values must be the same type (pandas.Dataframe)")
        else:
            self.__inner_data[self.y_pred_col_name] = y_pred_values.iloc[:, 0]
        self.df_contains = 'xyypred'
        self.max_x_column = -2

    def add_calculated_column(self, condition, on_column, new_column_name):
        """
        Adding a column at the end of the dataframe containing a calculated value with condition
        :param condition: string that represent a condition apply by lambda x: (ex : 'x > 3')
        :param on_column: string containing the name of the column to apply the condition
        :param new_column_name: string containing the name of the new column containing the values
        """
        columns = self.__inner_data.columns
        self.__inner_data[new_column_name] = self.__inner_data[on_column].map(eval('lambda x:' + condition))
        new_columns = columns.insert(self.max_x_column, new_column_name)
        self.__inner_data = self.__inner_data[new_columns]

    # deprecated
    # def add_column_in_data(self, new_column, column_name=None):
    #     """
    #     add a new column at the end of the data
    #     """
    #     if column_name is None:
    #         column_name = 'C' + str(len(self.__inner_data.columns) + self.max_x_column)
    #     self.__inner_data.insert(len(self.__inner_data.columns) + self.max_x_column, column_name, new_column)

    @property
    def x(self):
        if isinstance(self.df, pd.DataFrame):
            return self.__inner_data.iloc[:, 0:self.max_x_column].to_numpy()
        if isinstance(self.df, dd.DataFrame):
            return self.__inner_data.iloc[:, 0:self.max_x_column].to_dask_array(lengths=True)

    @property
    def y(self):
        if isinstance(self.df, pd.DataFrame):
            return self.__inner_data[self.y_col_name].to_numpy()
        if isinstance(self.df, dd.DataFrame):
            return self.__inner_data[self.y_col_name].to_dask_array(lengths=True)

    @property
    def y_pred(self):
        if isinstance(self.df, pd.DataFrame):
            return self.__inner_data[self.y_pred_col_name].to_numpy()
        if isinstance(self.df, dd.DataFrame):
            return self.__inner_data[self.y_pred_col_name].to_dask_array(lengths=True)

    @property
    def df(self):
        return self.__inner_data

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
    def from_dict(dictionary, df):
        """
        Transform a dictionary containing 'data.x' and 'data.y' to the input object
        :param dictionary: source dictionary
        :param df: dataframe containing data
        Raise mlsurvey.exception.ModelError if dictionary does not contain all keys
        """
        try:
            result = Data(df,
                          dictionary['df_contains'],
                          dictionary['y_col_name'],
                          dictionary['y_pred_col_name']
                          )
        except KeyError as e:
            raise mls.exceptions.ModelError(e)
        return result

    def merge_all(self):
        """
        transform the dataframe to an array containing all values
        Only for a pandas dataframe.
        :return: numpy.array or dask.array containing all data
        """
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

    def copy_with_new_data(self, data_array):
        """
        create a new instance of data with the same parameters but with different data values
        :param data_array: new data values as numpy array
        :return: an instance of data
        """
        data_array = np.concatenate((data_array[0], np.array([data_array[1]]).T), axis=1)
        create_function = None
        if isinstance(self.df, pd.DataFrame):
            create_function = pd.DataFrame
        if isinstance(self.df, dd.DataFrame):
            create_function = dd.from_array
        df = create_function(data_array, columns=self.df.columns)
        data = mls.models.Data(df,
                               df_contains=self.df_contains,
                               y_col_name=self.y_col_name)
        return data

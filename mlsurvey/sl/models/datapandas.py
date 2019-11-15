import dask.dataframe as dd
import numpy as np
import pandas as pd

import mlsurvey as mls
from .data import Data
from .data_factory import DataFactory


class DataPandas(Data):

    def set_pred_data(self, y_pred_values, y_pred_col_name=None):
        """
        set the y predicted data with values. see mls.models.Data.set_pred_data()
        Raise an exception if y_pred_values id not a pandas dataframe
        """
        if isinstance(self._inner_data, pd.DataFrame) and isinstance(y_pred_values, dd.DataFrame):
            raise mls.exceptions.ModelError("__inner_data and y_pred_values must be the same type (pandas.Dataframe)")
        else:
            super().set_pred_data(y_pred_values, y_pred_col_name)

    @property
    def x(self):
        return self._inner_data.iloc[:, 0:self.max_x_column].to_numpy()

    @property
    def y(self):
        return self._inner_data[self.y_col_name].to_numpy()

    @property
    def y_pred(self):
        return self._inner_data[self.y_pred_col_name].to_numpy()

    @staticmethod
    def from_dict(dictionary, df):
        """ see mls.models.Data.from_dict() """
        try:
            result = DataPandas(df,
                                dictionary['df_contains'],
                                dictionary['y_col_name'],
                                dictionary['y_pred_col_name']
                                )
        except KeyError as e:
            raise mls.exceptions.ModelError(e)
        return result

    def copy_with_new_data(self, data_array):
        """  see mls.models.Data.copy_with_new_data() """
        data_array = np.concatenate((data_array[0], np.array([data_array[1]]).T), axis=1)
        df = pd.DataFrame(data_array, columns=self.df.columns)
        data = mls.sl.models.DataPandas(df,
                                        df_contains=self.df_contains,
                                        y_col_name=self.y_col_name)
        return data

    class Factory:
        @staticmethod
        def create(df, df_contains='xy', y_col_name=None, y_pred_col_name=None):
            return DataPandas(df, df_contains, y_col_name, y_pred_col_name)

        @staticmethod
        def from_dict(d, df):
            return DataPandas.from_dict(d, df)


DataFactory.add_factory('Pandas', DataPandas.Factory)

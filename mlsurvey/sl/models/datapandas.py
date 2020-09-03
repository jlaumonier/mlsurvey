import numpy as np
import pandas as pd

import mlsurvey as mls
from .data import Data
from .data_factory import DataFactory


class DataPandas(Data):

    @property
    def x(self):
        return self._inner_data.iloc[:, 0:self.max_x_column].to_numpy()

    @property
    def x_df(self):
        return self._inner_data.iloc[:, 0:self.max_x_column]

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

    def copy_with_new_data_dataframe(self, dataframe):
        """  see mls.models.Data.copy_with_new_data_dataframe() """
        data = mls.sl.models.DataPandas(dataframe,
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

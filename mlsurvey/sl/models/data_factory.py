class DataFactory:
    factories = {}

    @staticmethod
    def add_factory(name, data_factory):
        """
        Add a DataFactory into all the factories
        :param name: the name of the factory
        :param data_factory: the instance of the factory
        """
        DataFactory.factories[name] = data_factory

    @staticmethod
    def create_data(storage, df, df_contains='xy', y_col_name=None, y_pred_col_name=None):
        """
        create a data from its storage type
        :param storage : the type of dataframe to store the data. By default 'Pandas' (only value at the moment)
        :param df : see mls.models.Data()
        :param df_contains : see mls.models.Data()
        :param y_col_name : see mls.models.Data()
        :param y_pred_col_name : see mls.models.Data()
        :return: the instance of the Data
        """
        return DataFactory.factories[storage].create(df, df_contains, y_col_name, y_pred_col_name)

    @staticmethod
    def create_data_from_dict(storage, d, df):
        """
        create a data from a dictionary and a dataframe
        :param storage : the type of dataframe to store the data. By default 'Pandas' (only value at the moment)
        :param d: the dictionary
        :param df: dataframe containing data
        :return: the instance of the dataset
        """
        data = DataFactory.factories[storage].from_dict(d, df)
        return data

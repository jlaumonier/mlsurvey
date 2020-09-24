# from sklearn.model_selection import train_test_split

import mlsurvey as mls
from mlsurvey.workflows.tasks import BaseTask


class SplitDataTask(BaseTask):
    """
    split data from prepared data  (train/test)
    """

    def requires(self):
        return [mls.sl.workflows.tasks.LoadDataTask(logging_directory=self.logging_directory,
                                                    logging_base_directory=self.logging_base_directory,
                                                    config_filename=self.config_filename,
                                                    config_directory=self.config_directory,
                                                    base_directory=self.base_directory),
                mls.sl.workflows.tasks.PrepareDataTask(logging_directory=self.logging_directory,
                                                       logging_base_directory=self.logging_base_directory,
                                                       config_filename=self.config_filename,
                                                       config_directory=self.config_directory,
                                                       base_directory=self.base_directory
                                                       )
                ]

    def run(self):
        """
        split the data for training/testing process.
        At the moment, only the split 'traintest' to split into train and test set is supported
        """
        loaded_data = self.log.load_input(self.input()[1]['data'].filename)
        data = loaded_data['data']
        loaded_raw_data = self.log.load_input(self.input()[0]['raw_data'].filename)
        raw_data = loaded_raw_data['raw_data']

        split_name = self.config.data['learning_process']['parameters']['split']
        split_param = self.config.data['splits'][split_name]['parameters']
        if self.config.data['splits'][split_name]['type'] == 'traintest':
            # TODO test shuffle False
            if split_param['shuffle']:
                df_test = data.df.sample(frac=split_param['test_size']/len(data.df),
                                         random_state=split_param['random_state'])
            else:
                df_test = data.df.head(len(data.df) * split_name['test_size'])
            df_train = data.df.drop(df_test.index)

            data_train = data.copy_with_new_data_dataframe(df_train)
            data_test = data.copy_with_new_data_dataframe(df_test)
            raw_data_train_df = raw_data.df.iloc[data_train.df.index]
            raw_data_train = raw_data.copy_with_new_data_dataframe(raw_data_train_df)
            raw_data_test_df = raw_data.df.iloc[data_test.df.index]
            raw_data_test = raw_data.copy_with_new_data_dataframe(raw_data_test_df)

            # reindex
            data_train.df.reset_index(drop=True, inplace=True)
            data_test.df.reset_index(drop=True, inplace=True)
            raw_data_train.df.reset_index(drop=True, inplace=True)
            raw_data_test.df.reset_index(drop=True, inplace=True)

            data_to_save = {'train': data_train,
                            'test': data_test,
                            'raw_train': raw_data_train,
                            'raw_test': raw_data_test}
            self.log.save_input(data_to_save, metadata_filename=self.output()['split_data'].filename)

    def output(self):
        data_json_filename = 'split_data.json'
        target_data = mls.sl.workflows.tasks.FileDirLocalTarget(directory=self.log.directory,
                                                                filename=data_json_filename)
        target = {'split_data': target_data}

        return target

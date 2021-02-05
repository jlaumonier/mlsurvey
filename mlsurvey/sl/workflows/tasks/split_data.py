from kedro.pipeline import node

from mlsurvey.workflows.tasks import BaseTask


class SplitDataTask(BaseTask):
    """
    split data from prepared data  (train/test)
    """

    @classmethod
    def get_node(cls):
        return node(SplitDataTask.split_data,
                    inputs=['config', 'log', 'raw_data', 'prepared_data'],
                    outputs=['train_data', 'test_data', 'train_raw_data', 'test_raw_data'])

    @staticmethod
    def split_data(config, log, raw_data, prepared_data):
        """
        split the data for training/testing process.
        At the moment, only the split 'traintest' to split into train and test set is supported
        """
        split_params = config.data['learning_process']['parameters']['split']
        if split_params['type'] == 'traintest':
            # TODO test shuffle False
            if split_params['parameters']['shuffle']:
                df_test = prepared_data.df.sample(frac=split_params['parameters']['test_size'] / len(prepared_data.df),
                                                  random_state=split_params['parameters']['random_state'])
            else:
                df_test = prepared_data.df.head(len(prepared_data.df) * split_params['parameters']['test_size'])
            df_train = prepared_data.df.drop(df_test.index)

            data_train = prepared_data.copy_with_new_data_dataframe(df_train)
            data_test = prepared_data.copy_with_new_data_dataframe(df_test)
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
            SplitDataTask.log_inputs_outputs(log, data_to_save)

            return [data_train, data_test, raw_data_train, raw_data_test]

    @classmethod
    def log_inputs_outputs(cls, log, d):
        # Log inside sub directory
        log.set_sub_dir(str(cls.__name__))
        inputs = {'train': d['train'],
                  'test': d['test'],
                  'raw_train': d['raw_train'],
                  'raw_test': d['raw_test']}
        log.save_input(inputs, metadata_filename='split_data.json')
        log.set_sub_dir('')



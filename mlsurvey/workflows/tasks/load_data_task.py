import os

from kedro.pipeline import node

import mlsurvey as mls


class LoadDataTask:
    """
    Load/generate raw data from dataset described in a config file
    """

    @staticmethod
    def init_dataset(dataset_dic):
        """
        Initialized dataset from config dictionary
        :param dataset_dic: dictionary describing the dataset
        :return: a Dataset object
        """
        dataset = mls.sl.datasets.DataSetFactory.create_dataset_from_dict(dataset_dic)
        return dataset

    @classmethod
    def log_inputs_outputs(cls, log, d):
        task_log = mls.Logging(base_dir=log.base_dir,
                               dir_name=os.path.join(log.dir_name,  str(cls.__name__)),
                               mlflow_run_id=log.mlflow_run_id)
        # log config
        task_log.log_config('config.json', d['config'].data)
        # log dataset
        task_log.save_dict_as_json('dataset.json', d['dataset'].to_dict())
        # log raw_data
        inputs = {'raw_data': d['raw_data']}
        task_log.save_input(inputs, 'raw_data.json')

    @staticmethod
    def load_data(config, log):
        # init dataset
        dataset_params_contents = config.data['learning_process']['parameters']['input']
        dataset = LoadDataTask.init_dataset(dataset_params_contents)

        # init data
        data = mls.sl.models.DataFactory.create_data(dataset.storage,
                                                     dataset.generate(),
                                                     y_col_name=dataset.metadata['y_col_name'])

        # TODO keep only some columns
        # if 'loading' in self.config.data['learning_process']['parameters']['input']:
        #     loading_params = self.config.data['learning_process']['parameters']['input']['loading']
        #     if 'columns_kept' in loading_params:
        #         columns_kept = list(loading_params['columns_kept'])
        #         raw_data = raw_data.copy_with_new_data_dataframe(raw_data.df[columns_kept])

        d = {'config': config,
             'dataset': dataset,
             'raw_data': data}
        LoadDataTask.log_inputs_outputs(log, d)

        return [dataset, data]

    @classmethod
    def get_node(cls):
        return node(LoadDataTask.load_data, inputs=['config', 'log'], outputs=['dataset', 'data'])

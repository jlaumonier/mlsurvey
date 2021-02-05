from kedro.pipeline import node

import mlsurvey as mls
from .base_task import BaseTask


class LoadDataTask(BaseTask):
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

        # log config
        log.log_config('config.json', d['config'].data)
        # Log inside sub directory
        log.set_sub_dir(str(cls.__name__))
        # log dataset
        log.save_dict_as_json('dataset.json', d['dataset'].to_dict())
        # log raw_data
        inputs = {'raw_data': d['raw_data']}
        log.save_input(inputs, 'raw_data.json')
        log.set_sub_dir('')

    @staticmethod
    def load_data(config, log, base_directory=''):
        # init dataset
        dataset_params_contents = config.data['learning_process']['parameters']['input']
        dataset = LoadDataTask.init_dataset(dataset_params_contents)

        # this line is only for FileDataSet testing...
        # Not sure if it is the most Pythonic and most TDDic way....
        if hasattr(dataset, 'set_base_directory'):
            dataset.set_base_directory(base_directory)

        # init data
        data = mls.sl.models.DataFactory.create_data(dataset.storage,
                                                     dataset.generate(),
                                                     y_col_name=dataset.metadata['y_col_name'])

        if 'loading' in config.data['learning_process']['parameters']['input']:
            loading_params = config.data['learning_process']['parameters']['input']['loading']
            if 'columns_kept' in loading_params:
                columns_kept = list(loading_params['columns_kept'])
                data = data.copy_with_new_data_dataframe(data.df[columns_kept])

        d = {'config': config,
             'dataset': dataset,
             'raw_data': data}
        LoadDataTask.log_inputs_outputs(log, d)

        return [dataset, data]

    @classmethod
    def get_node(cls):
        return node(LoadDataTask.load_data, inputs=['config', 'log', 'base_directory'], outputs=['dataset', 'raw_data'])

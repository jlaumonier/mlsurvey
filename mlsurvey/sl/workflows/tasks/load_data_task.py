import mlsurvey as mls
from mlsurvey.workflows.tasks import BaseTask


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

    def run(self):
        """
        Run the task
        """

        # init dataset
        dataset_params_contents = self.config.data['learning_process']['parameters']['input']
        dataset = self.init_dataset(dataset_params_contents)

        # this line is only for FileDataSet testing...
        # Not sure if it is the most Pythonic and most TDDic way....
        if hasattr(dataset, 'set_base_directory'):
            dataset.set_base_directory(self.base_directory)

        # init data
        raw_data = mls.sl.models.DataFactory.create_data(dataset.storage,
                                                         dataset.generate(),
                                                         y_col_name=dataset.metadata['y_col_name'])

        # keep only some columns
        if 'loading' in self.config.data['learning_process']['parameters']['input']:
            loading_params = self.config.data['learning_process']['parameters']['input']['loading']
            if 'columns_kept' in loading_params:
                columns_kept = list(loading_params['columns_kept'])
                raw_data = raw_data.copy_with_new_data_dataframe(raw_data.df[columns_kept])

        # log config
        self.log.log_config(self.output()['config'].filename, self.config.data)
        # log dataset
        self.log.save_dict_as_json(self.output()['dataset'].filename, dataset.to_dict())
        # log raw_data
        inputs = {'raw_data': raw_data}
        self.log.save_input(inputs, self.output()['raw_data'].filename)

    def output(self):
        dataset_json_filename = 'dataset.json'
        config_json_filename = 'config.json'
        raw_data_json_filename = 'raw_data.json'
        target_dataset = mls.sl.workflows.tasks.FileDirLocalTarget(directory=self.log.directory,
                                                                   filename=dataset_json_filename)
        target_config = mls.sl.workflows.tasks.FileDirLocalTarget(directory=self.log.directory,
                                                                  filename=config_json_filename)
        target_raw_data = mls.sl.workflows.tasks.FileDirLocalTarget(directory=self.log.directory,
                                                                    filename=raw_data_json_filename)
        target = {'config': target_config, 'dataset': target_dataset, 'raw_data': target_raw_data}
        return target

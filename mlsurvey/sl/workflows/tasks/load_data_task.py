import mlsurvey as mls
from mlsurvey.sl.workflows.tasks import SupervisedLearningTask


class LoadDataTask(SupervisedLearningTask):
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
        dataset_name = self.config.data['learning_process']['input']
        dataset_params_contents = self.config.data['datasets'][dataset_name]
        dataset = self.init_dataset(dataset_params_contents)

        # init data
        raw_data = mls.sl.models.DataFactory.create_data(dataset.storage,
                                                         dataset.generate(),
                                                         y_col_name=dataset.metadata['y_col_name'])

        # log config
        self.log.save_dict_as_json(self.output()['config'].filename, self.config.data)
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

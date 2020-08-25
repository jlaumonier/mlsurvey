import luigi

import mlsurvey as mls


class LoadDataTask(luigi.Task):
    """
    Load/generate raw data from dataset described in a config file
    """
    logging_base_directory = luigi.Parameter()
    logging_directory = luigi.Parameter()
    config_directory = luigi.Parameter()
    config_filename = luigi.Parameter()
    log = None
    config = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # init log and config
        self.init_log_config()

    @staticmethod
    def init_dataset(dataset_dic):
        """
        Initialized dataset from config dictionary
        :param dataset_dic: dictionary describing the dataset
        :return: a Dataset object
        """
        dataset = mls.sl.datasets.DataSetFactory.create_dataset_from_dict(dataset_dic)
        return dataset

    def init_log_config(self):
        """
        Initialized log and config
        """
        self.log = mls.Logging(dir_name=self.logging_directory, base_dir=self.logging_base_directory)
        self.config = mls.Config(name=self.config_filename, directory=self.config_directory)

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

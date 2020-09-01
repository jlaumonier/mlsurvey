from sklearn.model_selection import train_test_split

import mlsurvey as mls
from mlsurvey.workflows.tasks import BaseTask


class SplitDataTask(BaseTask):
    """
    split data from prepared data  (train/test)
    """

    def requires(self):
        return mls.sl.workflows.tasks.PrepareDataTask(logging_directory=self.logging_directory,
                                                      logging_base_directory=self.logging_base_directory,
                                                      config_filename=self.config_filename,
                                                      config_directory=self.config_directory)

    def run(self):
        """
        split the data for training/testing process.
        At the moment, only the split 'traintest' to split into train and test set is supported
        """
        loaded_data = self.log.load_input(self.input()['data'].filename)
        data = loaded_data['data']

        split_name = self.config.data['learning_process']['split']
        split_param = self.config.data['splits'][split_name]['parameters']
        if self.config.data['splits'][split_name]['type'] == 'traintest':
            (data_train_x,
             data_test_x,
             data_train_y,
             data_test_y) = train_test_split(data.x,
                                             data.y,
                                             test_size=split_param['test_size'],
                                             random_state=split_param['random_state'],
                                             shuffle=split_param['shuffle'])
            data_train = data.copy_with_new_data([data_train_x, data_train_y])
            data_test = data.copy_with_new_data([data_test_x, data_test_y])

            data_to_save = {'train': data_train, 'test': data_test}
            self.log.save_input(data_to_save, metadata_filename=self.output()['split_data'].filename)

    def output(self):
        data_json_filename = 'split_data.json'
        target_data = mls.sl.workflows.tasks.FileDirLocalTarget(directory=self.log.directory,
                                                                filename=data_json_filename)
        target = {'split_data': target_data}

        return target

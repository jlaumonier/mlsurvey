import mlsurvey as mls
from mlsurvey.workflows.tasks import BaseTask


class LearnTask(BaseTask):
    """
    learn model from train data
    """

    def requires(self):
        return mls.sl.workflows.tasks.SplitDataTask(logging_directory=self.logging_directory,
                                                    logging_base_directory=self.logging_base_directory,
                                                    config_filename=self.config_filename,
                                                    config_directory=self.config_directory,
                                                    base_directory=self.base_directory)

    def run(self):
        loaded_data = self.log.load_input(self.input()['split_data'].filename)
        data_train = loaded_data['train']

        algorithm_params = self.config.data['learning_process']['parameters']['algorithm']
        algorithm = mls.sl.models.Algorithm(algorithm_params)
        classifier = algorithm.learn(data_train.x, data_train.y)
        self.log.save_dict_as_json(self.output()['algorithm'].filename, algorithm.to_dict())
        self.log.save_classifier(classifier, filename=self.output()['model'].filename)

    def output(self):
        algorithm_filename = 'algorithm.json'
        algorithm = mls.sl.workflows.tasks.FileDirLocalTarget(directory=self.log.directory,
                                                              filename=algorithm_filename)
        model_filename = 'model.joblib'
        model = mls.sl.workflows.tasks.FileDirLocalTarget(directory=self.log.directory,
                                                          filename=model_filename)
        target = {'algorithm': algorithm, 'model': model}
        return target

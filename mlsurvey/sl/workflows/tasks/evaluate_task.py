import luigi
from sklearn.metrics import confusion_matrix

import mlsurvey as mls
from mlsurvey.workflows.tasks import BaseTask


class EvaluateTask(BaseTask):
    """
    evaluate model from test data
    """
    evaluation_type = luigi.Parameter()

    def requires(self):
        return [mls.sl.workflows.tasks.LoadDataTask(logging_directory=self.logging_directory,
                                                    logging_base_directory=self.logging_base_directory,
                                                    config_filename=self.config_filename,
                                                    config_directory=self.config_directory),
                mls.sl.workflows.tasks.SplitDataTask(logging_directory=self.logging_directory,
                                                     logging_base_directory=self.logging_base_directory,
                                                     config_filename=self.config_filename,
                                                     config_directory=self.config_directory),
                mls.sl.workflows.tasks.LearnTask(logging_directory=self.logging_directory,
                                                 logging_base_directory=self.logging_base_directory,
                                                 config_filename=self.config_filename,
                                                 config_directory=self.config_directory)
                ]

    def run(self):
        dataset_dict = self.log.load_json_as_dict(self.input()[0]['dataset'].filename)
        dataset = mls.sl.datasets.DataSetFactory.create_dataset_from_dict(dataset_dict)
        loaded_data = self.log.load_input(self.input()[1]['split_data'].filename)
        data_test = loaded_data['test']
        classifier = self.log.load_classifier(self.input()[2]['model'].filename)

        func_create_df = mls.Utils.func_create_dataframe(dataset.storage)
        df = func_create_df(classifier.predict(data_test.x), columns=['target_pred'])
        data_test.set_pred_data(df)

        evaluation = eval(str(self.evaluation_type) + '()')
        evaluation.score = classifier.score(data_test.x, data_test.y)
        evaluation.confusion_matrix = confusion_matrix(data_test.y, data_test.y_pred)
        self.log.save_dict_as_json(self.output()['evaluation'].filename, evaluation.to_dict())
        pass

    def output(self):
        evaluation_filename = 'evaluation.json'
        evaluation = mls.sl.workflows.tasks.FileDirLocalTarget(directory=self.log.directory,
                                                               filename=evaluation_filename)

        target = {'evaluation': evaluation}
        return target

import numpy as np
from sklearn.metrics import confusion_matrix

import mlsurvey as mls
from mlsurvey.workflows.tasks import BaseTask


class EvaluateTask(BaseTask):
    """
    evaluate model from test data
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
                                                       base_directory=self.base_directory),
                mls.sl.workflows.tasks.SplitDataTask(logging_directory=self.logging_directory,
                                                     logging_base_directory=self.logging_base_directory,
                                                     config_filename=self.config_filename,
                                                     config_directory=self.config_directory,
                                                     base_directory=self.base_directory),
                mls.sl.workflows.tasks.LearnTask(logging_directory=self.logging_directory,
                                                 logging_base_directory=self.logging_base_directory,
                                                 config_filename=self.config_filename,
                                                 config_directory=self.config_directory,
                                                 base_directory=self.base_directory)
                ]

    def run(self):
        dataset_dict = self.log.load_json_as_dict(self.input()[0]['dataset'].filename)
        dataset = mls.sl.datasets.DataSetFactory.create_dataset_from_dict(dataset_dict)
        loaded_data = self.log.load_input(self.input()[2]['split_data'].filename)
        data_test = loaded_data['test']

        classifier = self.log.load_classifier(self.input()[3]['model'].filename)

        func_create_df = mls.Utils.func_create_dataframe(dataset.storage)
        df = func_create_df(classifier.predict(data_test.x), columns=['target_pred'])
        data_test.set_pred_data(df)

        evaluation = mls.sl.models.EvaluationSupervised()
        evaluation.score = classifier.score(data_test.x, data_test.y)
        evaluation.confusion_matrix = confusion_matrix(data_test.y, data_test.y_pred)

        # Fairness
        if dataset.fairness:

            loaded_raw_data = self.log.load_input(self.input()[0]['raw_data'].filename)
            raw_data = loaded_raw_data['raw_data']
            loaded_prepared_data = self.log.load_input(self.input()[1]['data'].filename)
            prepared_data = loaded_prepared_data['data']

            func_create_df = mls.Utils.func_create_dataframe(dataset.storage)
            df_prepared_data = func_create_df(classifier.predict(prepared_data.x), columns=['target_pred'])
            raw_data.set_pred_data(df_prepared_data)

            # demographic parity
            column_str = raw_data.df.columns[dataset.fairness['protected_attribute']]
            raw_data.add_calculated_column(dataset.fairness['privileged_classes'], column_str, 'priv_class')

            # Demographic parity
            # P(Y=1 | Priv_class = False) - P(Y=1 | Priv_class = True)
            false_proba = mls.FairnessUtils.calculate_cond_probability(raw_data, [('target', 1)],
                                                                       [('priv_class', False)])
            true_proba = mls.FairnessUtils.calculate_cond_probability(raw_data, [('target', 1)],
                                                                      [('priv_class', True)])

            sub_evaluation = mls.sl.models.EvaluationFairness()
            sub_evaluation.demographic_parity = false_proba - true_proba
            # Disparate impact ratio
            # P(Y=1 | Priv_class = False) / P(Y=1 | Priv_class = True)
            sub_evaluation.disparate_impact_rate = false_proba / true_proba

            # criteria that use the classier
            if data_test.df_contains == 'xyypred' and not np.isnan(raw_data.y_pred).any():
                # equal opportunity
                # P(Y_hat=0 | Y=1, Priv_class = False) - P(Y_hat=0 | Y=1, Priv_class = True)
                false_proba = mls.FairnessUtils.calculate_cond_probability(raw_data, [('target_pred', 0)],
                                                                           [('target', 1), ('priv_class', False)])
                true_proba = mls.FairnessUtils.calculate_cond_probability(raw_data, [('target_pred', 0)],
                                                                          [('target', 1), ('priv_class', True)])
                sub_evaluation.equal_opportunity = false_proba - true_proba

                # statistical parity
                # P(Y_hat=1 | Priv_class = False) - P(y_hat=1 | Priv_class = True)
                false_proba = mls.FairnessUtils.calculate_cond_probability(raw_data, [('target_pred', 1)],
                                                                           [('priv_class', False)])
                true_proba = mls.FairnessUtils.calculate_cond_probability(raw_data, [('target_pred', 1)],
                                                                          [('priv_class', True)])
                sub_evaluation.statistical_parity = false_proba - true_proba

                # average equalized odds
                # Sum_i\inI [P(Y_hat=1 | Y=i, Priv_class = False) - P(Y_hat=1 | Y=i, Priv_class = True)] / |I|
                diff = 0
                for i in [0, 1]:
                    false_proba = mls.FairnessUtils.calculate_cond_probability(raw_data, [('target_pred', 1)],
                                                                               [('target', i), ('priv_class', False)])
                    true_proba = mls.FairnessUtils.calculate_cond_probability(raw_data, [('target_pred', 1)],
                                                                              [('target', i), ('priv_class', True)])
                    diff = diff + (false_proba - true_proba)
                sub_evaluation.average_equalized_odds = diff / 2.0
            evaluation.sub_evaluation = sub_evaluation

        self.log.save_dict_as_json(self.output()['evaluation'].filename, evaluation.to_dict())

    def output(self):
        evaluation_filename = 'evaluation.json'
        evaluation = mls.sl.workflows.tasks.FileDirLocalTarget(directory=self.log.directory,
                                                               filename=evaluation_filename)

        target = {'evaluation': evaluation}
        return target

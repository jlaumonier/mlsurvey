from sklearn.metrics import confusion_matrix

from kedro.pipeline import node

import mlsurvey as mls
from mlsurvey.workflows.tasks import BaseTask


class EvaluateTask(BaseTask):
    """
    evaluate model from test data
    """

    @classmethod
    def _log_inputs_outputs(cls, log, d):
        log.set_sub_dir(str(cls.__name__))
        # TODO Redondant ?
        if log.is_log_to_mlflow:
            log.mlflow_client.log_metric(log.mlflow_run.info.run_id, 'score', d['evaluation'].score)
        log.log_metrics('evaluation.json', d['evaluation'].to_dict())
        log.set_sub_dir('')

    @staticmethod
    def evaluate(config, log, dataset, raw_data, prepared_data, test_data, model_fullpath):

        classifier = log.load_classifier(model_fullpath, name_is_full_path=True)

        func_create_df = mls.Utils.func_create_dataframe(dataset.storage)
        df = func_create_df(classifier.predict(test_data.x), columns=['target_pred'])
        test_data.set_pred_data(df)

        evaluation = mls.sl.models.EvaluationSupervised()
        evaluation.score = classifier.score(test_data.x, test_data.y)
        evaluation.confusion_matrix = confusion_matrix(test_data.y, test_data.y_pred)

        # Fairness
        if dataset.fairness:
            value_target_is_one = dataset.fairness['target_is_one']
            value_target_is_zero = dataset.fairness['target_is_zero']

            func_create_df = mls.Utils.func_create_dataframe(dataset.storage)
            df_prepared_data = func_create_df(classifier.predict(prepared_data.x), columns=['target_pred'])
            df_prepared_data = df_prepared_data.replace(1.0, value_target_is_one)
            df_prepared_data = df_prepared_data.replace(0.0, value_target_is_zero)

            raw_data.set_pred_data(df_prepared_data)

            # demographic parity
            column_str = raw_data.df.columns[dataset.fairness['protected_attribute']]
            raw_data.add_calculated_column(dataset.fairness['privileged_classes'], column_str, 'priv_class')

            # Demographic parity
            # P(Y=1 | Priv_class = False) - P(Y=1 | Priv_class = True)
            false_proba = mls.FairnessUtils.calculate_cond_probability(raw_data,
                                                                       [(raw_data.y_col_name, value_target_is_one)],
                                                                       [('priv_class', False)])
            true_proba = mls.FairnessUtils.calculate_cond_probability(raw_data,
                                                                      [(raw_data.y_col_name, value_target_is_one)],
                                                                      [('priv_class', True)])

            sub_evaluation = mls.sl.models.EvaluationFairness()
            sub_evaluation.demographic_parity = false_proba - true_proba
            # Disparate impact ratio
            # P(Y=1 | Priv_class = False) / P(Y=1 | Priv_class = True)
            sub_evaluation.disparate_impact_rate = false_proba / true_proba

            # criteria that use the classifier
            if test_data.df_contains == 'xyypred':
                # equal opportunity
                # P(Y_hat=0 | Y=1, Priv_class = False) - P(Y_hat=0 | Y=1, Priv_class = True)
                false_proba = mls.FairnessUtils.calculate_cond_probability(raw_data,
                                                                           [('target_pred', value_target_is_zero)],
                                                                           [(raw_data.y_col_name, value_target_is_one),
                                                                            ('priv_class', False)])
                true_proba = mls.FairnessUtils.calculate_cond_probability(raw_data,
                                                                          [('target_pred', value_target_is_zero)],
                                                                          [(raw_data.y_col_name, value_target_is_one),
                                                                           ('priv_class', True)])
                sub_evaluation.equal_opportunity = false_proba - true_proba

                # statistical parityPrepareDataTask
                # P(Y_hat=1 | Priv_class = False) - P(y_hat=1 | Priv_class = True)
                false_proba = mls.FairnessUtils.calculate_cond_probability(raw_data,
                                                                           [('target_pred', value_target_is_one)],
                                                                           [('priv_class', False)])
                true_proba = mls.FairnessUtils.calculate_cond_probability(raw_data,
                                                                          [('target_pred', value_target_is_one)],
                                                                          [('priv_class', True)])
                sub_evaluation.statistical_parity = false_proba - true_proba

                # average equalized odds
                # Sum_i\inI [P(Y_hat=1 | Y=i, Priv_class = False) - P(Y_hat=1 | Y=i, Priv_class = True)] / |I|
                diff = 0
                for i in [value_target_is_zero, value_target_is_one]:
                    false_proba = mls.FairnessUtils.calculate_cond_probability(raw_data,
                                                                               [('target_pred', value_target_is_one)],
                                                                               [(raw_data.y_col_name, i),
                                                                                ('priv_class', False)])
                    true_proba = mls.FairnessUtils.calculate_cond_probability(raw_data,
                                                                              [('target_pred', value_target_is_one)],
                                                                              [(raw_data.y_col_name, i),
                                                                               ('priv_class', True)])
                    diff = diff + (false_proba - true_proba)
                sub_evaluation.average_equalized_odds = diff / 2.0
            evaluation.sub_evaluation = sub_evaluation

        d = {'evaluation': evaluation}
        EvaluateTask._log_inputs_outputs(log, d)
        return [evaluation]

    @classmethod
    def get_node(cls):
        return node(EvaluateTask.evaluate,
                    inputs=['config', 'log', 'dataset', 'raw_data', 'prepared_data', 'test_data', 'model_fullpath'],
                    outputs=['evaluation'])


import numpy as np

import mlsurvey as mls
from .learning_workflow import LearningWorkflow


class FairnessWorkflow(LearningWorkflow):

    def __init__(self, config_file='config_fairness.json', config_directory='config/', base_directory='', context=None):
        super().__init__(config_directory=config_directory, base_directory=base_directory)
        if context is None:
            self.config = mls.Config(config_file, directory=self.config_directory)
        else:
            self.config = None
        self.is_sub_process = (context is not None)
        self.parent_context = context
        self.context = mls.models.Context(eval_type=mls.models.EvaluationFairness)
        self.log = mls.Logging()
        self.task_terminated_get_data = False
        self.task_terminated_evaluate = False
        self.task_terminated_persistence = False

    def set_terminated(self):
        """ set the workflow as terminated if all tasks are terminated"""
        self.terminated = (self.task_terminated_get_data
                           & self.task_terminated_evaluate
                           & self.task_terminated_persistence)

    def set_subprocess_terminated(self):
        """ set the subprocess workflow as terminated if all tasks are terminated except persistence"""
        self.terminated = (self.task_terminated_get_data
                           & self.task_terminated_evaluate)

    def task_get_data(self):
        """
        Initialize and generate the dataset from configuration learning_process.input
        """
        try:
            if not self.is_sub_process:
                dataset_name = self.config.data['fairness_process']['input']
                dataset_params = self.config.data['datasets'][dataset_name]['parameters']
                dataset_type = self.config.data['datasets'][dataset_name]['type']
                self.context.dataset = mls.datasets.DataSetFactory.create_dataset(dataset_type)

                # this line is only for FileDataSet testing...
                # Not sure if it is the most Pythonic and most TDDic way....
                if hasattr(self.context.dataset, 'set_base_directory'):
                    self.context.dataset.set_base_directory(self.base_directory)

                self.context.dataset.set_generation_parameters(dataset_params)
                dataset_fairness = self.config.data['datasets'][dataset_name]['fairness']
                self.context.dataset.set_fairness_parameters(dataset_fairness)
                self.context.data = mls.models.Data(self.context.dataset.generate())
            else:
                self.context.dataset = self.parent_context.dataset
                if not self.context.dataset.fairness:
                    raise mls.exceptions.WorkflowError("Fairness parameters are required")
                self.context.data = self.parent_context.raw_data

            self.task_terminated_get_data = True
        except KeyError as e:
            raise mls.exceptions.ConfigError(e)

    def task_evaluate(self):
        """ calculate the fairness of the data."""
        # demographic parity
        column_str = self.context.data.df.columns[self.context.dataset.fairness['protected_attribute']]
        self.context.data.add_calculated_column(self.context.dataset.fairness['privileged_classes'],
                                                column_str,
                                                'priv_class')

        # Demographic parity
        # P(Y=1 | Priv_class = False) - P(Y=1 | Priv_class = True)
        false_proba = mls.FairnessUtils.calculate_cond_probability(self.context.data, [('target', 1)],
                                                                   [('priv_class', False)])
        true_proba = mls.FairnessUtils.calculate_cond_probability(self.context.data, [('target', 1)],
                                                                  [('priv_class', True)])
        self.context.evaluation.demographic_parity = false_proba - true_proba
        # Disparate impact ratio
        # P(Y=1 | Priv_class = False) / P(Y=1 | Priv_class = True)
        self.context.evaluation.disparate_impact_rate = false_proba / true_proba

        # criteria that use the classier
        if self.context.data.df_contains == 'xyypred' and not np.isnan(self.context.data.y_pred).any():
            # equal opportunity
            # P(Y_hat=0 | Y=1, Priv_class = False) - P(Y_hat=0 | Y=1, Priv_class = True)
            false_proba = mls.FairnessUtils.calculate_cond_probability(self.context.data, [('target_pred', 0)],
                                                                       [('target', 1), ('priv_class', False)])
            true_proba = mls.FairnessUtils.calculate_cond_probability(self.context.data, [('target_pred', 0)],
                                                                      [('target', 1), ('priv_class', True)])
            self.context.evaluation.equal_opportunity = false_proba - true_proba

            # statistical parity
            # P(Y_hat=1 | Priv_class = False) - P(y_hat=1 | Priv_class = True)
            false_proba = mls.FairnessUtils.calculate_cond_probability(self.context.data, [('target_pred', 1)],
                                                                       [('priv_class', False)])
            true_proba = mls.FairnessUtils.calculate_cond_probability(self.context.data, [('target_pred', 1)],
                                                                      [('priv_class', True)])
            self.context.evaluation.statistical_parity = false_proba - true_proba

            # average equalized odds
            # Sum_i\inI [P(Y_hat=1 | Y=i, Priv_class = False) - P(Y_hat=1 | Y=i, Priv_class = True)] / |I|
            diff = 0
            for i in [0, 1]:
                false_proba = mls.FairnessUtils.calculate_cond_probability(self.context.data, [('target_pred', 1)],
                                                                           [('target', i), ('priv_class', False)])
                true_proba = mls.FairnessUtils.calculate_cond_probability(self.context.data, [('target_pred', 1)],
                                                                          [('target', i), ('priv_class', True)])
                diff = diff + (false_proba - true_proba)
            self.context.evaluation.average_equalized_odds = diff / 2.0

        self.task_terminated_evaluate = True

    def task_persist(self):
        """
        save all aspects of the learning into files (config, data sets, classifier, evaluation)
        """
        self.log.save_dict_as_json('config.json', self.config.data)
        self.context.save(self.log)
        self.task_terminated_persistence = True

    def run(self):
        """
        Run all tasks
            - get and generate data
            - evaluate
            - save all aspects
        """
        self.task_get_data()
        self.task_evaluate()
        self.task_persist()
        self.set_terminated()

    def run_as_subprocess(self):
        """
        Run the following tasks
            - get and generate data
            - evaluate
        """
        self.task_get_data()
        self.task_evaluate()
        self.set_subprocess_terminated()

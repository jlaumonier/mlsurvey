import numpy as np

import mlsurvey as mls
from .learning_workflow import LearningWorkflow


class FairnessWorkflow(LearningWorkflow):

    def __init__(self, config_file='config_fairness.json', config_directory='config/', base_directory=''):
        super().__init__(config_directory=config_directory, base_directory=base_directory)
        self.config = mls.Config(config_file, directory=self.config_directory)
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

    def task_get_data(self):
        """
        Initialize and generate the dataset from configuration learning_process.input
        """
        try:
            dataset_name = self.config.data['fairness_process']['input']
            dataset_params = self.config.data['datasets'][dataset_name]['parameters']
            dataset_type = self.config.data['datasets'][dataset_name]['type']
            self.context.dataset = mls.datasets.DataSetFactory.create_dataset(dataset_type)

            # this line is only for FileDataSet testing... Not sure if it is the most Pythonic and most TDDic way....
            if hasattr(self.context.dataset, 'set_base_directory'):
                self.context.dataset.set_base_directory(self.base_directory)

            self.context.dataset.set_generation_parameters(dataset_params)
            dataset_fairness = self.config.data['datasets'][dataset_name]['fairness']
            self.context.dataset.set_fairness_parameters(dataset_fairness)
            self.context.data.x, self.context.data.y = self.context.dataset.generate()
            self.task_terminated_get_data = True
        except KeyError as e:
            raise mls.exceptions.ConfigError(e)

    def task_evaluate(self):
        """ calculate the fairness of the data. Does nothing at the moment :S. Fairness to implement"""
        # demographic parity
        column = self.context.dataset.fairness['protected_attribute']
        privileged_selected = np.array(list(map(eval('lambda x:' + self.context.dataset.fairness['privileged_classes']),
                                                self.context.data.x[:, column])))
        extended_data = self.context.data.copy()
        # number of column in data
        n = self.context.data.x.shape[1]
        # number of examples in data
        m = self.context.data.x.shape[0]
        extended_data.x = np.concatenate((extended_data.x, privileged_selected.reshape(-1, m).T), axis=1)
        # privileged_data = mls.models.Data()
        # unprivileged_data = mls.models.Data()
        # privileged_data.x = self.context.data.x[privileged_selected]
        # privileged_data.y = self.context.data.y[privileged_selected]
        # unprivileged_data.x = self.context.data.x[~privileged_selected]
        # unprivileged_data.y = self.context.data.y[~privileged_selected]
        proba = mls.FairnessUtils.calculate_all_cond_probability(extended_data)
        self.context.evaluation.demographic_parity = proba[n]['0.0'][1] - proba[n]['1.0'][1]
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

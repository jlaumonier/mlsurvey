from multiprocessing import Pool

import mlsurvey as mls
from .learning_workflow import LearningWorkflow


class MultipleLearningWorkflow(LearningWorkflow):

    def __init__(self, config_file='config.json', config_directory='config/'):
        """
        Initialized the multiple learning workflow
        :param config_file: config file for initializing the workflow, Used if config is None
        """
        super().__init__(config_directory=config_directory)
        self.task_terminated_expand_config = False
        self.task_terminated_run_each_config = False
        try:
            self.config = mls.Config(config_file, directory=self.config_directory)
        except (FileNotFoundError, mls.exceptions.ConfigError):
            self.task_terminated_init = False
        self.expanded_config = []
        self.slw = []

    def set_terminated(self):
        """ set the workflow as terminated if all tasks are terminated"""
        self.terminated = (self.task_terminated_init
                           & self.task_terminated_expand_config
                           & self.task_terminated_run_each_config)

    def task_expand_config(self):
        """
        Get a configuration file with lists in parameters and hyperparameter for datasets, algorithms and splits
        and generate a list containing the cartesian product of all possible configs.
        :return: a list of dictionary, each one is a config usable by a supervised learning workflow
        """
        for lp_element in list(mls.Utils.dict_generator_cartesian_product(self.config.data['learning_process'])):
            input_parameters = self.config.data['datasets'][lp_element['input']]['parameters']
            for ds_element in list(mls.Utils.dict_generator_cartesian_product(input_parameters)):
                one_dataset = dict()
                one_dataset[lp_element['input']] = self.config.data['datasets'][lp_element['input']].copy()
                one_dataset[lp_element['input']]['parameters'] = ds_element
                algorithm_hyperparameters = self.config.data['algorithms'][lp_element['algorithm']]['hyperparameters']
                for alg_element in list(mls.Utils.dict_generator_cartesian_product(algorithm_hyperparameters)):
                    one_algo = dict()
                    one_algo[lp_element['algorithm']] = self.config.data['algorithms'][lp_element['algorithm']].copy()
                    one_algo[lp_element['algorithm']]['hyperparameters'] = alg_element
                    split_parameters = self.config.data['splits'][lp_element['split']]['parameters']
                    for split_element in list(mls.Utils.dict_generator_cartesian_product(split_parameters)):
                        one_split = dict()
                        one_split[lp_element['split']] = self.config.data['splits'][lp_element['split']].copy()
                        one_split[lp_element['split']]['parameters'] = split_element
                        one_config = {'datasets': one_dataset,
                                      'algorithms': one_algo,
                                      'splits': one_split,
                                      'learning_process': lp_element}
                        self.expanded_config.append(one_config)
        self.task_terminated_expand_config = True

    @staticmethod
    def task_run_one_config(c):
        """
        Run one supervised learning workflow
        :param c: configuration for the supervised learning workflow
        :return: the supervised learning workflow after a run
        """
        sl = mls.workflows.SupervisedLearningWorkflow(config=c)
        sl.run()
        return sl

    def task_run_each_config(self):
        """
        Run each config with a supervised learning workflow generated by task_expand_config()
        """
        results = []
        pool = Pool()
        for c in self.expanded_config:
            results.append(pool.apply_async(MultipleLearningWorkflow.task_run_one_config, (c,)))
        pool.close()
        pool.join()
        for res in results:
            self.slw.append(res.get())
        self.task_terminated_run_each_config = True

    def run(self):
        """
        Run the workflow only if the initialization was done correctly:
        - Expand configs
        - run each config
        - terminated the workflow
        """
        if self.task_terminated_init:
            self.task_expand_config()
            self.task_run_each_config()
            self.set_terminated()

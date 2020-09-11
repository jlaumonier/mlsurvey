import mlsurvey as mls
from mlsurvey.workflows.tasks import BaseTask


class ExpandConfigTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expanded_config = self.expand_config()

    def expand_config(self):
        """
        Get a configuration file with lists in parameters and hyperparameter for datasets, algorithms and splits
        and generate a list containing the cartesian product of all possible configs.
        :return: a list of dictionary, each one is a config usable by a supervised learning workflow
        """
        expanded_config = []
        for lp_element in list(mls.Utils.dict_generator_cartesian_product(self.config.data['learning_process'])):
            input_parameters = self.config.data['datasets'][lp_element['input']]['parameters']
            if 'fairness' in self.config.data['datasets'][lp_element['input']]:
                fairness_parameters = self.config.data['datasets'][lp_element['input']]['fairness']
            else:
                fairness_parameters = {}
            for ds_element in list(mls.Utils.dict_generator_cartesian_product(input_parameters)):
                for fp_element in list(mls.Utils.dict_generator_cartesian_product(fairness_parameters)):
                    one_dataset = dict()
                    one_dataset[lp_element['input']] = self.config.data['datasets'][lp_element['input']].copy()
                    one_dataset[lp_element['input']]['parameters'] = ds_element
                    if fairness_parameters != {}:
                        one_dataset[lp_element['input']]['fairness'] = fp_element
                    algorithm_hyperparameters = self.config.data['algorithms'][lp_element['algorithm']][
                        'hyperparameters']
                    for alg_element in list(mls.Utils.dict_generator_cartesian_product(algorithm_hyperparameters)):
                        one_algo = dict()
                        one_algo[lp_element['algorithm']] = self.config.data['algorithms'][
                            lp_element['algorithm']].copy()
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
                            expanded_config.append(one_config)
        return expanded_config

    def run(self):
        for id_config, config in enumerate(self.expanded_config):
            self.log.save_dict_as_json(self.output()['expanded_config'][id_config].filename, config)
        self.log.save_dict_as_json(self.output()['config'].filename, self.config.data)

    def output(self):
        targets_expanded_configs = []
        for id_config, config in enumerate(self.expanded_config):
            config_json_filename = 'expand_config'+str(id_config).zfill(len(str(len(self.expanded_config))))+'.json'
            target_config = mls.sl.workflows.tasks.FileDirLocalTarget(directory=self.log.directory,
                                                                      filename=config_json_filename)
            targets_expanded_configs.append(target_config)
        multiple_config_json_filename = 'config.json'
        config = mls.sl.workflows.tasks.FileDirLocalTarget(directory=self.log.directory,
                                                           filename=multiple_config_json_filename)
        target = {'config': config, 'expanded_config': targets_expanded_configs}
        return target

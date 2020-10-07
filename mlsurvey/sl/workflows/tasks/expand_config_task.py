import mlsurvey as mls
from mlsurvey.workflows.tasks import BaseTask


class ExpandConfigTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expanded_config = mls.ExpandDict.run(self.config.data)

    def run(self):
        for id_config, config in enumerate(self.expanded_config):
            self.log.save_dict_as_json(self.output()['expanded_config'][id_config].filename, config)
        self.log.save_dict_as_json(self.output()['config'].filename, self.config.data)

    def output(self):
        targets_expanded_configs = []
        for id_config, config in enumerate(self.expanded_config):
            config_json_filename = 'expand_config' + str(id_config).zfill(len(str(len(self.expanded_config)))) + '.json'
            target_config = mls.sl.workflows.tasks.FileDirLocalTarget(directory=self.log.directory,
                                                                      filename=config_json_filename)
            targets_expanded_configs.append(target_config)
        multiple_config_json_filename = 'config.json'
        config = mls.sl.workflows.tasks.FileDirLocalTarget(directory=self.log.directory,
                                                           filename=multiple_config_json_filename)
        target = {'config': config, 'expanded_config': targets_expanded_configs}
        return target

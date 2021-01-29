from kedro.pipeline import node

import mlsurvey as mls
from mlsurvey.workflows.tasks import BaseTask


class ExpandConfigTask(BaseTask):

    @staticmethod
    def expand_config(config, log):
        expanded_config = mls.ExpandDict.run(config.data)
        # save files
        for id_config, conf in enumerate(expanded_config):
            config_json_filename = 'expand_config' + str(id_config).zfill(len(str(len(expanded_config)))) + '.json'
            log.save_dict_as_json(config_json_filename, conf)
        log.save_dict_as_json('config.json', config.data)
        return expanded_config

    @classmethod
    def get_node(cls):
        return node(ExpandConfigTask.expand_config, inputs=['config', 'log'], outputs='expanded_config')

import unittest
import os

import mlsurvey as mls


class TaskTestCase(unittest.TestCase):

    @classmethod
    def _init_config_log(cls, config_filename, base_directory, config_directory):
        final_config_directory = os.path.join(str(base_directory), str(config_directory))
        config = mls.Config(name=config_filename, directory=final_config_directory)
        config.compact()
        # init logging
        log = mls.Logging(base_dir=os.path.join(base_directory, 'logs'),
                          mlflow_log=True)
        return config, log

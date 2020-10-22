import luigi
import os
import logging

import mlsurvey as mls


@luigi.Task.event_handler(luigi.Event.START)
def begin_task(task):
    task.log.msg(task.__class__.__name__ + ' is Start', logging.INFO)


@luigi.Task.event_handler(luigi.Event.PROGRESS)
def progress_task(params):
    task = params['task']
    msg = params['msg']
    task.log.msg(task.__class__.__name__ + ' progress : ' + msg, logging.INFO)


@luigi.Task.event_handler(luigi.Event.SUCCESS)
def end_task_success(task):
    task.log.msg(task.__class__.__name__ + ' is Done', logging.INFO)


class BaseTask(luigi.Task):

    base_directory = luigi.Parameter(default='')
    logging_base_directory = luigi.Parameter()
    logging_directory = luigi.Parameter()
    config_directory = luigi.Parameter(default='config/')
    config_filename = luigi.Parameter(default='config.json')
    log = None
    config = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # init log and config
        self.init_log_config()

    def init_log_config(self):
        """
        Initialized log and config
        """
        # TODO refactoring directory
        final_config_directory = os.path.join(str(self.base_directory), str(self.config_directory))
        self.log = mls.Logging(dir_name=self.logging_directory, base_dir=self.logging_base_directory)
        self.config = mls.Config(name=self.config_filename, directory=final_config_directory)
        self.config.compact()

    def run(self):
        self.trigger_event(luigi.Event.PROGRESS, {'task': self, 'msg': 'running'})


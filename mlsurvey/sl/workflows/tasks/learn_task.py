import os

from kedro.pipeline import node

import mlsurvey as mls
from mlsurvey.workflows.tasks import BaseTask


class LearnTask(BaseTask):
    """
    learn model from train data
    """
    @classmethod
    def _log_inputs_outputs(cls, log, d):
        log.set_sub_dir(str(cls.__name__))
        model_fullpathname = os.path.join(log.directory, 'model.joblib')
        # Log model metadata
        log.save_dict_as_json('model.json', d['model_metadata'])
        log.save_dict_as_json('algorithm.json', d['algorithm'].to_dict())
        log.save_classifier(d['model'], filename='model.joblib')
        log.set_sub_dir('')
        return model_fullpathname

    @staticmethod
    def learn(config, log, train_data):
        algorithm_params = config.data['learning_process']['parameters']['algorithm']
        algorithm = mls.sl.models.Algorithm(algorithm_params)
        classifier = algorithm.learn(train_data.x, train_data.y)

        # Logging
        metadata = {'type': config.data['learning_process']['parameters']['algorithm']['type']}
        d = {'model_metadata': metadata,
             'algorithm': algorithm,
             'model': classifier}
        model_fullpathname = LearnTask._log_inputs_outputs(log, d)
        return [model_fullpathname]

    @classmethod
    def get_node(cls):
        return node(LearnTask.learn, inputs=['config', 'log', 'train_data'], outputs=['model_fullpath'])


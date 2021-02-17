from kedro.pipeline import node
import mlsurvey as mls
from mlsurvey.workflows.tasks import BaseTask


class MultipleLearningTask(BaseTask):
    """
    learn model from train data
    """

    @staticmethod
    def task_run_one_config(p):
        """
        Run one supervised learning workflow
        :param p: tupple parameter containing
               c: configuration filename for the supervised learning workflow
               cd: directory of the config file
               bd: base directory for all file loaded with config
               wf_t: type of the workflow to launch
        :return: the workflow after a run
        """
        c = p[0]
        bd = p[1]
        wf_t = p[2]

        wf = wf_t(config_dict=c, base_directory=bd, mlflow_log=True)
        wf.run()
        return wf

    @staticmethod
    def run(config, log, expanded_config):
        wf_type_string = config.data['learning_process']['type']
        wf_type = mls.Utils.import_from_dotted_path(wf_type_string)
        slw = []
        configs_pandas = [(c, '', wf_type) for c in expanded_config]

        for c in configs_pandas:
            res_wf = MultipleLearningTask.task_run_one_config(c)
            slw.append(res_wf)

        result = {'NbLearning': len(slw)}
        log.save_dict_as_json('results.json', result)

    @classmethod
    def get_node(cls):
        return node(MultipleLearningTask.run,
                    inputs=['config', 'log', 'expanded_config'],
                    outputs=None)

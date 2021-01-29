from kedro.pipeline import node
import mlsurvey as mls
from mlsurvey.workflows.tasks import BaseTask
from tqdm import tqdm
from multiprocessing import Pool


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
        cd = p[1]
        bd = p[2]
        wf_t = p[3]

        wf = wf_t(config_file=c, config_directory=cd, base_directory=bd)
        wf.run()
        return wf

    @staticmethod
    def run(config, log, expanded_config):
        wf_type_string = config.data['learning_process']['type']
        wf_type = mls.Utils.import_from_dotted_path(wf_type_string)
        slw = []
        configs_pandas = [(c.filename, c.directory, self.base_directory, wf_type) for c in expanded_config]
        pool = Pool(processes=1)
        with tqdm(total=len(configs_pandas)) as pbar:
            enumr = enumerate(pool.imap_unordered(MultipleLearningTask.task_run_one_config,
                                                  configs_pandas))
            for i, res in tqdm(enumr):
                slw.append(res)
                pbar.update()
        pool.close()
        pool.join()

        result = {'NbLearning': len(slw)}
        log.save_dict_as_json('results.json', result)

    @classmethod
    def get_node(cls):
        return node(MultipleLearningTask.run, inputs=['config', 'log', 'expanded_config'], outputs=None)

import mlsurvey as mls
from mlsurvey.workflows.tasks import BaseTask
from tqdm import tqdm
from multiprocessing import Pool


class MultipleLearningTask(BaseTask):
    """
    learn model from train data
    """

    def requires(self):
        return mls.sl.workflows.tasks.ExpandConfigTask(logging_directory=self.logging_directory,
                                                       logging_base_directory=self.logging_base_directory,
                                                       config_filename=self.config_filename,
                                                       config_directory=self.config_directory,
                                                       base_directory=self.base_directory)

    @staticmethod
    def task_run_one_config(p):
        """
        Run one supervised learning workflow
        :param p: tupple parameter containing
               c: configuration filename for the supervised learning workflow
               cd: directory of the config file
               bd: base directory for all file loaded with config
        :return: the supervised learning workflow after a run
        """
        c = p[0]
        cd = p[1]
        bd = p[2]

        sl = mls.sl.workflows.SupervisedLearningWorkflow(config_file=c, config_directory=cd, base_directory=bd)
        sl.run()
        return sl

    def run(self):

        slw = []
        configs_pandas = [(c.filename, c.directory, self.base_directory) for c in self.input()]
        pool = Pool()
        with tqdm(total=len(configs_pandas)) as pbar:
            enumr = enumerate(pool.imap_unordered(MultipleLearningTask.task_run_one_config,
                                                  configs_pandas))
            for i, res in tqdm(enumr):
                slw.append(res)
                pbar.update()
        pool.close()
        pool.join()

        result = {'NbLearning': len(slw)}
        self.log.save_dict_as_json(self.output()['result'].filename, result)

        # for config in self.input():
        #     sl = mls.sl.workflows.SupervisedLearningWorkflow(config_file=config.filename,
        #                                                      config_directory=config.directory,
        #                                                      base_directory=self.base_directory)
        #     sl.run()

    def output(self):
        result_filename = 'results.json'
        result = mls.sl.workflows.tasks.FileDirLocalTarget(directory=self.log.directory,
                                                           filename=result_filename)

        target = {'result': result}
        return target

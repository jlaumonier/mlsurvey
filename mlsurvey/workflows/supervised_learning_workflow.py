from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import mlsurvey as mls
from .learning_workflow import LearningWorkflow


class SupervisedLearningWorkflow(LearningWorkflow):

    def __init__(self, config_file='config.json', config=None, config_directory='config/'):
        """
        Initialized the supervised learning workflow
        :param config_file: config file for initializing the workflow, Used if config is None
        :param config: dictionary for config. If set, replace config file
        """
        super().__init__()
        try:
            self.config = mls.Config(config_file, config=config, directory=config_directory)
        except (FileNotFoundError, mls.exceptions.ConfigError):
            self.task_terminated_init = False
        self.data_preparation = StandardScaler()
        self.context = mls.models.Context(eval_type=mls.models.EvaluationSupervised)
        self.log = mls.Logging()
        self.task_terminated_get_data = False
        self.task_terminated_prepare_data = False
        self.task_terminated_split_data = False
        self.task_terminated_learn = False
        self.task_terminated_evaluate = False
        self.task_terminated_persistence = False

    def set_terminated(self):
        """ set the workflow as terminated if all tasks are terminated"""
        self.terminated = (self.task_terminated_init
                           & self.task_terminated_get_data
                           & self.task_terminated_prepare_data
                           & self.task_terminated_split_data
                           & self.task_terminated_learn
                           & self.task_terminated_evaluate
                           & self.task_terminated_persistence)

    def task_get_data(self):
        """
        Initialize and generate the dataset from configuration learning_process.input
        """
        dataset_name = self.config.data['learning_process']['input']
        dataset_params = self.config.data['datasets'][dataset_name]['parameters']
        dataset_type = self.config.data['datasets'][dataset_name]['type']
        self.context.dataset = mls.datasets.DataSetFactory.create_dataset(dataset_type)
        self.context.dataset.set_generation_parameters(dataset_params)
        self.context.data.x, self.context.data.y = self.context.dataset.generate()
        self.task_terminated_get_data = True

    def task_prepare_data(self):
        """ Prepare the data. At that time, prepared with StandardScaler()"""
        self.context.data.x = self.data_preparation.fit_transform(self.context.data.x)
        self.task_terminated_prepare_data = True

    def task_split_data(self):
        """ split the data for training/testing process.
        At the moment, only the split 'traintest' to split into train and test set is supported
        """
        split_name = self.config.data['learning_process']['split']
        split_param = self.config.data['splits'][split_name]['parameters']
        if self.config.data['splits'][split_name]['type'] == 'traintest':
            (self.context.data_train.x,
             self.context.data_test.x,
             self.context.data_train.y,
             self.context.data_test.y) = train_test_split(self.context.data.x,
                                                          self.context.data.y,
                                                          test_size=split_param['test_size'],
                                                          random_state=split_param['random_state'],
                                                          shuffle=split_param['shuffle'])
            self.task_terminated_split_data = True

    def task_learn(self):
        """ learn the classifier with train data"""
        algorithm_name = self.config.data['learning_process']['algorithm']
        self.context.algorithm = mls.models.Algorithm(self.config.data['algorithms'][algorithm_name])
        self.context.classifier = self.context.algorithm.learn(self.context.data_train.x, self.context.data_train.y)
        self.task_terminated_learn = True

    def task_evaluate(self):
        """ calculate the score of the classifier with test data """
        self.context.evaluation.score = self.context.classifier.score(self.context.data_test.x,
                                                                      self.context.data_test.y)
        self.task_terminated_evaluate = True

    def task_persist(self):
        """
        save all aspects of the learning into files (config, data sets, classifier, evaluation)
        """
        self.log.save_dict_as_json('config.json', self.config.data)
        self.context.save(self.log)
        self.task_terminated_persistence = True

    def run(self):
        """
        Run all tasks
            - get and generate data
            - prepare data
            - split data
            - learn
            - evaluate
            - save all aspects
        """
        self.task_get_data()
        self.task_prepare_data()
        self.task_split_data()
        self.task_learn()
        self.task_evaluate()
        self.task_persist()
        self.set_terminated()

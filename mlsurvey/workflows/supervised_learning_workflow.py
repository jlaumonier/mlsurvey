from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import mlsurvey as mls
from .learning_workflow import LearningWorkflow


class SupervisedLearningWorkflow(LearningWorkflow):

    def __init__(self, config_file='config.json', config=None):
        """
        Initialized the supervised learning workflow
        :param config_file: config file for initializing the workflow, Used if config is None
        :param config: dictionary for config. If set, replace config file
        """
        super().__init__()
        self.config = mls.Config(config_file, config=config)
        self.task_terminated_get_data = False
        self.task_terminated_prepare_data = False
        self.task_terminated_split_data = False
        self.task_terminated_learn = False
        self.task_terminated_evaluate = False
        self.task_terminated_persistence = False
        self.data = mls.datasets.DataSet()
        self.data_preparation = StandardScaler()
        self.data_train = mls.Input()
        self.data_test = mls.Input()
        self.algorithm = None
        self.classifier = None
        self.score = 0.0
        self.log = mls.Logging()

    def set_terminated(self):
        """ set the workflow as terminated if all tasks are terminated"""
        self.terminated = self.task_terminated_get_data \
                          & self.task_terminated_prepare_data \
                          & self.task_terminated_split_data \
                          & self.task_terminated_learn \
                          & self.task_terminated_evaluate \
                          & self.task_terminated_persistence

    def task_get_data(self):
        """
        Initialize and generate the dataset from configuration learning_process.input
        """
        dataset_name = self.config.data.learning_process.input
        data_params = self.config.data.datasets[dataset_name].parameters
        self.data = mls.datasets.DataSetFactory.create_dataset(self.config.data.datasets[dataset_name].type)
        self.data.set_generation_parameters(data_params)
        self.data.generate()
        self.task_terminated_get_data = True

    def task_prepare_data(self):
        """ Prepare the data. At that time, prepared with StandardScaler()"""
        self.data.x = self.data_preparation.fit_transform(self.data.x)
        self.task_terminated_prepare_data = True

    def task_split_data(self):
        """ split the data for training/testing process.
        At the moment, only the split 'traintest' to split into train and test set is supported
        """
        split_name = self.config.data.learning_process.split
        split_param = self.config.data.splits[split_name].parameters
        if self.config.data.splits[split_name].type == 'traintest':
            self.data_train.x, \
            self.data_test.x, \
            self.data_train.y, \
            self.data_test.y = train_test_split(self.data.x,
                                                self.data.y,
                                                test_size=split_param.test_size,
                                                random_state=split_param.random_state,
                                                shuffle=split_param.shuffle)
            self.task_terminated_split_data = True

    def task_learn(self):
        """ learn the classifier with train data"""
        algorithm_name = self.config.data.learning_process.algorithm
        algorithm = mls.Algorithm(self.config.data.algorithms[algorithm_name])
        self.classifier = algorithm.learn(self.data_train.x, self.data_train.y)
        self.task_terminated_learn = True

    def task_evaluate(self):
        """ calculate the score of the classifier with test data """
        self.score = self.classifier.score(self.data_test.x, self.data_test.y)
        self.task_terminated_evaluate = True

    def task_persist(self):
        """
        save all aspects of the learning into files (config, data sets, classifier, evaluation)
        """
        self.log.save_dict_as_json('config.json', self.config.data)
        inputs = {'train': self.data_train, 'test': self.data_test}
        self.log.save_input(inputs)
        self.log.save_classifier(self.classifier)
        evaluation = {'score': self.score}
        self.log.save_dict_as_json('evaluation.json', evaluation)
        self.task_terminated_persistence = True

    def load_data_classifier(self, directory):
        """
        load all aspects of the learning from files (config, data sets, classifier, evaluation)
        :param directory: directory where the files are stored
        """
        self.config = mls.Config('config.json', directory)
        self.log = mls.Logging(directory, base_dir='')
        inputs = self.log.load_input('input.json')
        self.data_train = inputs['train']
        self.data_test = inputs['test']
        self.classifier = self.log.load_classifier()
        evaluation = self.log.load_json_as_dict('evaluation.json')
        self.score = evaluation['score']

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

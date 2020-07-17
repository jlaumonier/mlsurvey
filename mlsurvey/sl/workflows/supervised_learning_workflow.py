from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import mlsurvey as mls
from mlsurvey.workflows.learning_workflow import LearningWorkflow


class SupervisedLearningWorkflow(LearningWorkflow):

    def __init__(self,
                 config_file='config.json',
                 config=None,
                 config_directory='config/',
                 base_directory='',
                 logging_dir=None):
        """
        Initialized the supervised learning workflow
        :param config_file: config file for initializing the workflow, Used if config is None
        :param config: dictionary for config. If set, replace config file
        """
        super().__init__(config_directory=config_directory, base_directory=base_directory)
        try:
            self.config = mls.Config(config_file, config=config, directory=self.config_directory)
        except (FileNotFoundError, mls.exceptions.ConfigError):
            self.task_terminated_init = False
        self.data_preparation = StandardScaler()
        self.context = mls.sl.models.Context(eval_type=mls.sl.models.EvaluationSupervised)
        self.log = mls.Logging(dir_name=logging_dir)
        self.task_terminated_get_data = False
        self.task_terminated_prepare_data = False
        self.task_terminated_split_data = False
        self.task_terminated_learn = False
        self.task_terminated_evaluate = False
        self.task_terminated_fairness = False
        self.task_terminated_persistence = False

    def set_terminated(self):
        """ set the workflow as terminated if all tasks are terminated"""
        self.terminated = (self.task_terminated_init
                           & self.task_terminated_get_data
                           & self.task_terminated_prepare_data
                           & self.task_terminated_split_data
                           & self.task_terminated_learn
                           & self.task_terminated_evaluate
                           & self.task_terminated_fairness
                           & self.task_terminated_persistence)

    def task_get_data(self):
        """
        Initialize and generate the dataset from configuration learning_process.input
        """
        dataset_name = self.config.data['learning_process']['input']
        dataset_params = self.config.data['datasets'][dataset_name]['parameters']
        dataset_type = self.config.data['datasets'][dataset_name]['type']
        dataset_storage = 'Pandas'
        dataset_metadata = None
        if 'storage' in self.config.data['datasets'][dataset_name]:
            dataset_storage = self.config.data['datasets'][dataset_name]['storage']
        if 'metadata' in self.config.data['datasets'][dataset_name]:
            dataset_metadata = self.config.data['datasets'][dataset_name]['metadata']
        self.context.dataset = mls.sl.datasets.DataSetFactory.create_dataset(dataset_type, dataset_storage)
        self.context.dataset.set_generation_parameters(dataset_params)
        self.context.dataset.set_metadata_parameters(dataset_metadata)

        # this line is only for FileDataSet testing... Not sure if it is the most Pythonic and most TDDic way....
        if hasattr(self.context.dataset, 'set_base_directory'):
            self.context.dataset.set_base_directory(self.base_directory)

        if 'fairness' in self.config.data['datasets'][dataset_name]:
            dataset_fairness = self.config.data['datasets'][dataset_name]['fairness']
            self.context.dataset.set_fairness_parameters(dataset_fairness)

        self.context.raw_data = mls.sl.models.DataFactory.create_data(self.context.dataset.storage,
                                                                      self.context.dataset.generate(),
                                                                      y_col_name=self.context.dataset.metadata[
                                                                          'y_col_name'])
        self.task_terminated_get_data = True

    def task_prepare_data(self):
        """ Prepare the data. At that time, prepared with StandardScaler()"""
        x_transformed = self.data_preparation.fit_transform(self.context.raw_data.x)
        self.context.data = self.context.raw_data.copy_with_new_data([x_transformed, self.context.raw_data.y])
        self.task_terminated_prepare_data = True

    def task_split_data(self):
        """ split the data for training/testing process.
        At the moment, only the split 'traintest' to split into train and test set is supported
        """
        split_name = self.config.data['learning_process']['split']
        split_param = self.config.data['splits'][split_name]['parameters']
        if self.config.data['splits'][split_name]['type'] == 'traintest':
            (data_train_x,
             data_test_x,
             data_train_y,
             data_test_y) = train_test_split(self.context.data.x,
                                             self.context.data.y,
                                             test_size=split_param['test_size'],
                                             random_state=split_param['random_state'],
                                             shuffle=split_param['shuffle'])
            self.context.data_train = self.context.data.copy_with_new_data([data_train_x, data_train_y])
            self.context.data_test = self.context.data.copy_with_new_data([data_test_x, data_test_y])
            self.task_terminated_split_data = True

    def task_learn(self):
        """ learn the classifier with train data"""
        algorithm_name = self.config.data['learning_process']['algorithm']
        self.context.algorithm = mls.sl.models.Algorithm(self.config.data['algorithms'][algorithm_name])
        self.context.classifier = self.context.algorithm.learn(self.context.data_train.x, self.context.data_train.y)
        self.task_terminated_learn = True

    def task_evaluate(self):
        """ calculate the score of the classifier with test data """
        self.context.evaluation.score = self.context.classifier.score(self.context.data_test.x,
                                                                      self.context.data_test.y)
        func_create_df = mls.Utils.func_create_dataframe(self.context.dataset.storage)
        df = func_create_df(self.context.classifier.predict(self.context.data.x), columns=['target_pred'])
        self.context.data.set_pred_data(df)
        # Assuming that data and raw_data are the same data but transformed
        df = func_create_df(self.context.data.y_pred, columns=['target_pred'])
        self.context.raw_data.set_pred_data(df)
        df = func_create_df(self.context.classifier.predict(self.context.data_train.x), columns=['target_pred'])
        self.context.data_train.set_pred_data(df)
        df = func_create_df(self.context.classifier.predict(self.context.data_test.x), columns=['target_pred'])
        self.context.data_test.set_pred_data(df)
        if self.context.dataset.storage == 'Pandas':
            self.context.evaluation.confusion_matrix = confusion_matrix(self.context.data_test.y,
                                                                        self.context.data_test.y_pred)
        self.task_terminated_evaluate = True

    def task_fairness(self):
        if self.context.dataset.fairness:
            fw = mls.sl.workflows.FairnessWorkflow(context=self.context)
            fw.run_as_subprocess()
            self.context.evaluation.sub_evaluation = fw.context.evaluation
        self.task_terminated_fairness = True

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
        self.task_fairness()
        self.task_persist()
        self.set_terminated()

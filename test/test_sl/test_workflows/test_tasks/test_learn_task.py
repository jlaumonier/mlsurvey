import os
import shutil

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner
from kedro.pipeline.node import Node


from sklearn import neighbors
import mlflow

import mlsurvey as mls


class TestLearnTask(mls.testing.TaskTestCase):
    config_directory = ''
    base_directory = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.base_directory = os.path.join(directory, '../../../')
        cls.config_directory = os.path.join(cls.base_directory, 'config/')
        cls.mlflow_client = mlflow.tracking.MlflowClient()
        cls.mlflow_experiments = cls.mlflow_client.list_experiments()

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(os.path.join(cls.base_directory, log.base_dir), ignore_errors=True)

    def test_get_node(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LearnTask.get_node()
        :condition : -
        :main_result : create a kedro with input and output parameter
        """
        prepare_data_node = mls.sl.workflows.tasks.LearnTask.get_node()
        self.assertIsInstance(prepare_data_node, Node)

    def test_learn(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LearnTask.run()
        :condition : data file are split, saved in hdf database and logged
        :main_result : model is trained
        """
        config, log = self._init_config_log('complete_config_loaded.json',
                                            self.base_directory,
                                            self.config_directory)
        df_train_data = mls.FileOperation.read_hdf('train-content.h5',
                                                   os.path.join(self.base_directory, 'files/tasks/split_data'),
                                                   'Pandas')
        train_data = mls.sl.models.DataFactory.create_data('Pandas', df_train_data)
        [model_fullpath] = mls.sl.workflows.tasks.LearnTask.learn(config, log, train_data)
        self.assertTrue(os.path.isfile(model_fullpath))
        log.set_sub_dir(str(mls.sl.workflows.tasks.LearnTask.__name__))
        classifier = log.load_classifier()
        self.assertIsInstance(classifier, neighbors.KNeighborsClassifier)
        self.assertEqual(30, classifier.get_params()['leaf_size'])

    def _run_one_task(self, config_filename):
        # create node from Task
        load_data_node = mls.workflows.tasks.LoadDataTask.get_node()
        prepare_data_node = mls.sl.workflows.tasks.PrepareDataTask.get_node()
        split_data_node = mls.sl.workflows.tasks.SplitDataTask.get_node()
        learn_data_node = mls.sl.workflows.tasks.LearnTask.get_node()
        config, log = self._init_config_log(config_filename, self.base_directory, self.config_directory)
        # Prepare a data catalog
        data_catalog = DataCatalog({'config': MemoryDataSet(),
                                    'log': MemoryDataSet(),
                                    'base_directory': MemoryDataSet()})
        data_catalog.save('config', config)
        data_catalog.save('log', log)
        data_catalog.save('base_directory', self.base_directory)
        # Assemble nodes into a pipeline
        pipeline = Pipeline([load_data_node, prepare_data_node, split_data_node, learn_data_node])
        # Create a runner to run the pipeline
        runner = SequentialRunner()
        # Run the pipeline
        runner.run(pipeline, data_catalog)
        return log, config, data_catalog

    def test_run(self):
        """
        :test : mlsurvey.sl.workflows.tasks.LearnTask.run()
        :condition : data file are split, saved in hdf database and logged
        :main_result : model is trained
        """
        config_filename = 'complete_config_loaded.json'
        log, config, data_catalog = self._run_one_task(config_filename)
        log.set_sub_dir(str(mls.sl.workflows.tasks.LearnTask.__name__))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'model.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'model.joblib')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'algorithm.json')))

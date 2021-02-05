import os
import shutil

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner
from kedro.pipeline.node import Node
import mlflow

import mlsurvey as mls


class TestMultipleLearningTask(mls.testing.TaskTestCase):
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
        :test : ccf.workflows.tasks.MultipleLearningTask.get_node()
        :condition : -
        :main_result : create a kedro with input and output parameter
        """
        multiple_learning_node = mls.sl.workflows.tasks.MultipleLearningTask.get_node()
        self.assertIsInstance(multiple_learning_node, Node)

    def _run_one_task(self, config_filename):
        # create node from Task
        expand_config_node = mls.sl.workflows.tasks.ExpandConfigTask.get_node()
        multiple_learning_node = mls.sl.workflows.tasks.MultipleLearningTask.get_node()
        # Prepare a data catalog
        config, log = self._init_config_log(config_filename,
                                            self.base_directory,
                                            self.config_directory)
        expanded_config = [{"input": {"type": "NClassRandomClassificationWithNoise",
                                      "parameters": {"n_samples": 100, "shuffle": True, "random_state": 0, "noise": 0}
                                      },
                            "split": {"type": "traintest",
                                      "parameters": {"test_size": 20, "random_state": 0, "shuffle": True}
                                      },
                            "algorithm": {"type": "sklearn.neighbors.KNeighborsClassifier",
                                          "hyperparameters": {
                                              "n_neighbors": 15,
                                              "algorithm": "auto",
                                              "weights": "uniform"
                                          }
                                          }
                            },
                           {"input": {"type": "make_circles",
                                      "parameters": {
                                          "n_samples": 100,
                                          "shuffle": True,
                                          "noise": 0,
                                          "random_state": 0,
                                          "factor": 0.3
                                      }},
                            "split": {"type": "traintest",
                                      "parameters": {"test_size": 20, "random_state": 0, "shuffle": True}
                                      },
                            "algorithm": {"type": "sklearn.neighbors.KNeighborsClassifier",
                                          "hyperparameters": {
                                              "n_neighbors": 15,
                                              "algorithm": "auto",
                                              "weights": "uniform"
                                          }
                                          }
                            },
                           {"input": {"type": "load_iris",
                                      "parameters": {}
                                      },
                            "split": {"type": "traintest",
                                      "parameters": {"test_size": 20, "random_state": 0, "shuffle": True}
                                      },
                            "algorithm": {"type": "sklearn.neighbors.KNeighborsClassifier",
                                          "hyperparameters": {
                                              "n_neighbors": 15,
                                              "algorithm": "auto",
                                              "weights": "uniform"
                                          }
                                          }
                            }]

        data_catalog = DataCatalog({'config': MemoryDataSet(),
                                    'log': MemoryDataSet(),
                                    'base_directory': MemoryDataSet(),
                                    'expanded_config': MemoryDataSet()})
        data_catalog.save('config', config)
        data_catalog.save('log', log)
        data_catalog.save('base_directory', self.base_directory)
        data_catalog.save('expanded_config', expanded_config)
        # Assemble nodes into a pipeline
        pipeline = Pipeline([expand_config_node, multiple_learning_node])
        # Create a runner to run the pipeline
        runner = SequentialRunner()
        # Run the pipeline
        runner.run(pipeline, data_catalog)
        return log, data_catalog

    def test_run(self):
        """
        :test : mlsurvey.sl.workflows.tasks.TestMultipleLearningTask.run()
        :condition : config file contains multiple learning config
        :main_result : all learning have ran
        """
        log, data_catalog = self._run_one_task('multiple_config.json')
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'results.json')))
        result_dict = log.load_json_as_dict('results.json')
        self.assertEqual(3, result_dict['NbLearning'])

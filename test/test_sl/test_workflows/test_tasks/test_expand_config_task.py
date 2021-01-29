import os
import unittest
import shutil

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner

import mlflow

import mlsurvey as mls


class TestExpandConfigTask(unittest.TestCase):
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

    def _run_one_task(self, config_filename):
        # create node from Task
        expand_config_node = mls.sl.workflows.tasks.ExpandConfigTask.get_node()
        # Prepare a data catalog
        # TODO revoir gestion configuation et Logging
        final_config_directory = os.path.join(str(self.base_directory), str(self.config_directory))
        config = mls.Config(name=config_filename, directory=final_config_directory)
        config.compact()
        # init logging
        log = mls.Logging(base_dir=os.path.join(self.base_directory, 'logs'),
                          mlflow_log=True)

        data_catalog = DataCatalog({'config': MemoryDataSet(),
                                    'log': MemoryDataSet(),
                                    'expanded_config': MemoryDataSet()})
        data_catalog.save('config', config)
        data_catalog.save('log', log)
        # Assemble nodes into a pipeline
        pipeline = Pipeline([expand_config_node])
        # Create a runner to run the pipeline
        runner = SequentialRunner()
        # Run the pipeline
        runner.run(pipeline, data_catalog)
        return log, data_catalog

    def test_run_input_should_have_expanded(self):
        """
        :test : mlsurvey.sl.workflows.tasks.ExpandConfigTask.run()
        :condition : config file contains multiple values
        :main_result : configs have been expanded
        """
        log, data_catalog = self._run_one_task('multiple_config.json')
        expanded_config = data_catalog.load('expanded_config')

        self.assertTrue(os.path.isfile(os.path.join(log.base_dir, log.dir_name, 'config.json')))
        self.assertEqual('f12c72eb6037c48a634dbb2a0ae6e193',
                         mls.Utils.md5_file(os.path.join(log.directory, 'config.json')))
        list_files = [name for name in os.listdir(log.directory) if os.path.isfile(os.path.join(log.directory, name))]
        list_files = list(filter(lambda x: x.startswith('expand_config'), list_files))  # keeps only the expanded config
        list_files.sort()
        nb_files = len(list_files)
        self.assertEqual(3, nb_files)
        d = [{"input": {"type": "NClassRandomClassificationWithNoise",
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
        configs = []
        for id_file, file in enumerate(list_files):
            configs.append(mls.Config(file, directory=log.directory))
            self.assertDictEqual(d[id_file], expanded_config[id_file]['learning_process']['parameters'])
            self.assertDictEqual(d[id_file], configs[id_file].data['learning_process']['parameters'])

    def test_run_input_all_should_have_expanded(self):
        """
        :test : mlsurvey.sl.workflows.tasks.ExpandConfigTask.run()
        :condition : config file contains multiple values (more values)
        :main_result : configs have been expanded
        """
        log, data_catalog = self._run_one_task('full_multiple_config.json')

        list_files = [name for name in os.listdir(log.directory) if os.path.isfile(os.path.join(log.directory, name))]
        list_files = list(filter(lambda x: x.startswith('expand_config'), list_files))  # keeps only the expanded config
        list_files.sort()
        nb_files = len(list_files)
        self.assertEqual(72, nb_files)
        ds0 = {"type": "make_classification",
               "parameters": {"n_samples": 100,
                              "shuffle": True,
                              "noise": 0,
                              "random_state": 0
                              }
               }
        al32 = {"type": "svm", "hyperparameters": {"kernel": "rbf", "C": 1.0}}
        configs = []
        for id_file, file in enumerate(list_files):
            configs.append(mls.Config(file, directory=log.directory))
        self.assertDictEqual(ds0, configs[0].data['learning_process']['parameters']['input'])
        self.assertDictEqual(al32, configs[32].data['learning_process']['parameters']['algorithm'])
        self.assertIsInstance(configs[0].data['learning_process']['parameters']['input'], dict)

    def test_task_expand_config_fairness_should_have_expanded(self):
        """
        :test : mlsurvey.sl.workflows.tasks.ExpandConfigTask.run()
        :condition : config file contains lists in fairness parameters
        :main_result : should expand
        """
        log, data_catalog = self._run_one_task('multiple_config_multiple_fairness.json')

        list_files = [name for name in os.listdir(log.directory) if os.path.isfile(os.path.join(log.directory, name))]
        list_files = list(filter(lambda x: x.startswith('expand_config'), list_files))  # keeps only the expanded config
        list_files.sort()
        nb_files = len(list_files)
        self.assertEqual(2, nb_files)
        f1 = {"type": "FileDataSet",
              "parameters": {
                  "directory": "files/dataset",
                  "filename": "test-fairness.arff"
              },
              "fairness": {
                  "protected_attribute": 1,
                  "privileged_classes": "x >= 35"
              }
              }
        configs = []
        for id_file, file in enumerate(list_files):
            configs.append(mls.Config(file, directory=log.directory))
        self.assertDictEqual(f1, configs[1].data['learning_process']['parameters']['input'])
        self.assertIsInstance(configs[0].data['learning_process']['parameters']['input'], dict)

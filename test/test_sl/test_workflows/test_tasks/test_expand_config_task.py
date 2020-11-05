import os
import unittest
import shutil

import luigi
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

    def test_run_input_should_have_expanded(self):
        """
        :test : mlsurvey.sl.workflows.tasks.ExpandConfigTask.run()
        :condition : config file contains multiple values
        :main_result : configs have been expanded
        """
        temp_log = mls.Logging()
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        luigi.build([mls.sl.workflows.tasks.ExpandConfigTask(logging_directory=temp_log.dir_name,
                                                             logging_base_directory=os.path.join(self.base_directory,
                                                                                                 temp_log.base_dir),
                                                             config_filename='multiple_config.json',
                                                             config_directory=self.config_directory,
                                                             base_directory=self.base_directory,
                                                             mlflow_run_id=run.info.run_id)], local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
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
            self.assertDictEqual(d[id_file], configs[id_file].data['learning_process']['parameters'])

    def test_run_input_all_should_have_expanded(self):
        """
        :test : mlsurvey.sl.workflows.tasks.ExpandConfigTask.run()
        :condition : config file contains multiple values (more values)
        :main_result : configs have been expanded
        """
        temp_log = mls.Logging()
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        luigi.build([mls.sl.workflows.tasks.ExpandConfigTask(logging_directory=temp_log.dir_name,
                                                             logging_base_directory=os.path.join(self.base_directory,
                                                                                                 temp_log.base_dir),
                                                             config_filename='full_multiple_config.json',
                                                             config_directory=self.config_directory,
                                                             base_directory=self.base_directory,
                                                             mlflow_run_id=run.info.run_id)], local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
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
        temp_log = mls.Logging()
        run = self.mlflow_client.create_run(self.mlflow_experiments[0].experiment_id)
        luigi.build([mls.sl.workflows.tasks.ExpandConfigTask(logging_directory=temp_log.dir_name,
                                                             logging_base_directory=os.path.join(self.base_directory,
                                                                                                 temp_log.base_dir),
                                                             config_filename='multiple_config_multiple_fairness.json',
                                                             config_directory=self.config_directory,
                                                             base_directory=self.base_directory,
                                                             mlflow_run_id=run.info.run_id)], local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
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

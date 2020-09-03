import os
import unittest
import shutil

import luigi

import mlsurvey as mls


class TestExpandConfigTask(unittest.TestCase):
    config_directory = ''
    base_directory = ''

    @classmethod
    def setUpClass(cls):
        directory = os.path.dirname(__file__)
        cls.base_directory = os.path.join(directory, '../../../')
        cls.config_directory = os.path.join(cls.base_directory, 'config/')

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
        luigi.build([mls.sl.workflows.tasks.ExpandConfigTask(logging_directory=temp_log.dir_name,
                                                             logging_base_directory=os.path.join(self.base_directory,
                                                                                                 temp_log.base_dir),
                                                             config_filename='multiple_config.json',
                                                             config_directory=self.config_directory,
                                                             base_directory=self.base_directory)], local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        list_files = [name for name in os.listdir(log.directory) if os.path.isfile(os.path.join(log.directory, name))]
        list_files.sort()
        nb_files = len(list_files)
        self.assertEqual(3, nb_files)
        d = [{"input": "DataSet1", "split": "traintest20", "algorithm": "knn-base"},
             {"input": "DataSet2", "split": "traintest20", "algorithm": "knn-base"},
             {"input": "DataSet3", "split": "traintest20", "algorithm": "knn-base"}]
        configs = []
        for id_file, file in enumerate(list_files):
            configs.append(mls.Config(file, directory=log.directory))
            self.assertDictEqual(d[id_file], configs[id_file].data['learning_process'])
        self.assertEqual('NClassRandomClassificationWithNoise', configs[0].data['datasets']['DataSet1']['type'])
        self.assertEqual(100, configs[0].data['datasets']['DataSet1']['parameters']['n_samples'])
        self.assertEqual('make_circles', configs[1].data['datasets']['DataSet2']['type'])
        self.assertEqual(100, configs[1].data['datasets']['DataSet2']['parameters']['n_samples'])
        self.assertEqual('load_iris', configs[2].data['datasets']['DataSet3']['type'])
        self.assertEqual(0, len(configs[2].data['datasets']['DataSet3']['parameters']))

    def test_run_input_all_should_have_expanded(self):
        """
        :test : mlsurvey.sl.workflows.tasks.ExpandConfigTask.run()
        :condition : config file contains multiple values (more values)
        :main_result : configs have been expanded
        """
        temp_log = mls.Logging()
        luigi.build([mls.sl.workflows.tasks.ExpandConfigTask(logging_directory=temp_log.dir_name,
                                                             logging_base_directory=os.path.join(self.base_directory,
                                                                                                 temp_log.base_dir),
                                                             config_filename='full_multiple_config.json',
                                                             config_directory=self.config_directory,
                                                             base_directory=self.base_directory)], local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        list_files = [name for name in os.listdir(log.directory) if os.path.isfile(os.path.join(log.directory, name))]
        list_files.sort()
        nb_files = len(list_files)
        self.assertEqual(72, nb_files)
        lp0 = {"input": "DataSet1", "split": "traintest20", "algorithm": "knn-base"}
        ds0 = {"type": "make_classification",
               "parameters": {
                   "n_samples": 100,
                   "shuffle": True,
                   "noise": 0,
                   "random_state": 0
               }
               }
        al32 = {"algorithm-family": "sklearn.neural_network.MLPClassifier",
                "hyperparameters": {
                    "hidden_layer_sizes": (1, 2, 3)}
                }
        configs = []
        for id_file, file in enumerate(list_files):
            configs.append(mls.Config(file, directory=log.directory))
        self.assertDictEqual(lp0, configs[0].data['learning_process'])
        self.assertDictEqual(ds0, configs[0].data['datasets']['DataSet1'])
        self.assertDictEqual(al32, configs[32].data['algorithms']['nn-multiple-layer-choice'])
        self.assertEqual(1, len(configs[0].data['datasets']))

    def test_task_expand_config_fairness_should_have_expanded(self):
        """
        :test : mlsurvey.sl.workflows.tasks.ExpandConfigTask.run()
        :condition : config file contains lists in fairness parameters
        :main_result : should expand
        """
        temp_log = mls.Logging()
        luigi.build([mls.sl.workflows.tasks.ExpandConfigTask(logging_directory=temp_log.dir_name,
                                                             logging_base_directory=os.path.join(self.base_directory,
                                                                                                 temp_log.base_dir),
                                                             config_filename='multiple_config_multiple_fairness.json',
                                                             config_directory=self.config_directory,
                                                             base_directory=self.base_directory)], local_scheduler=True)
        log = mls.Logging(base_dir=os.path.join(self.base_directory, temp_log.base_dir), dir_name=temp_log.dir_name)
        list_files = [name for name in os.listdir(log.directory) if os.path.isfile(os.path.join(log.directory, name))]
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
        self.assertDictEqual(f1, configs[1].data['datasets']['DataSetGermanCredit'])
        self.assertEqual(1, len(configs[0].data['datasets']))

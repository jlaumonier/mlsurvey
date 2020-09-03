import os
import shutil
import unittest
import uuid

import numpy as np
import sklearn.neighbors as neighbors

import mlsurvey as mls


class TestContext(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir, ignore_errors=True)

    def test_init_all_init(self):
        context = mls.sl.models.Context(eval_type=mls.sl.models.Evaluation)
        self.assertIsInstance(context.id, uuid.UUID)
        self.assertIsInstance(context.dataset, mls.sl.datasets.DataSet)
        self.assertIsNone(context.raw_data)
        self.assertIsNone(context.data)
        self.assertIsNone(context.data_train)
        self.assertIsNone(context.data_test)
        self.assertIsNone(context.algorithm)
        self.assertIsNone(context.classifier)
        self.assertIsInstance(context.evaluation, mls.sl.models.Evaluation)

    def test_init_context_supervised(self):
        context = mls.sl.models.Context(eval_type=mls.sl.models.EvaluationSupervised)
        self.assertIsInstance(context.evaluation, mls.sl.models.EvaluationSupervised)

    def test_init_unknown_eval_type(self):
        """
        :test : mlsurvey.models.Context()
        :condition : unknown evaluation
        :main_result : raise AttributeError
        """
        try:
            # This line makes a unresolved reference warning but that's what i want to test the context
            _ = mls.sl.models.Context(eval_type=mls.sl.models.UnknownEvaluation)
            self.assertTrue(False)
        except AttributeError:
            self.assertTrue(True)

    def test_save_context(self):
        context = mls.sl.models.Context(eval_type=mls.sl.models.EvaluationSupervised)
        config_algo = {
            'algorithm-family': 'sklearn.neighbors.KNeighborsClassifier',
            'hyperparameters': {
                'n_neighbors': 3,
                'algorithm': 'auto',
                'weights': 'uniform'
            }
        }
        context.algorithm = mls.sl.models.Algorithm(config_algo)
        log = mls.Logging()
        context.save(log)
        self.assertEqual(5, len(os.listdir(log.directory)))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'dataset.json')))
        self.assertEqual('1e0d460125a76c6990750a9cafb49d86',
                         mls.Utils.md5_file(os.path.join(log.directory, 'dataset.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'input.json')))
        self.assertEqual('e024075ecfdd447815a1226dc9eff25d',
                         mls.Utils.md5_file(os.path.join(log.directory, 'input.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'algorithm.json')))
        self.assertEqual('77ead84d71a01a8e83c8706b2b96cf57',
                         mls.Utils.md5_file(os.path.join(log.directory, 'algorithm.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'model.joblib')))
        self.assertEqual('4dc6000a33a1ca18a3aaedb7f9802955',
                         mls.Utils.md5_file(os.path.join(log.directory, 'model.joblib')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'evaluation.json')))
        self.assertEqual('c8a8c328f655178bdfd600a7710cf8e1',
                         mls.Utils.md5_file(os.path.join(log.directory, 'evaluation.json')))

    def test_save_context_no_algorithm_neither_classifier_should_save(self):
        context = mls.sl.models.Context(eval_type=mls.sl.models.EvaluationSupervised)
        log = mls.Logging()
        context.save(log)
        self.assertEqual(3, len(os.listdir(log.directory)))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'dataset.json')))
        self.assertEqual('1e0d460125a76c6990750a9cafb49d86',
                         mls.Utils.md5_file(os.path.join(log.directory, 'dataset.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'input.json')))
        self.assertEqual('e024075ecfdd447815a1226dc9eff25d',
                         mls.Utils.md5_file(os.path.join(log.directory, 'input.json')))
        self.assertTrue(os.path.isfile(os.path.join(log.directory, 'evaluation.json')))
        self.assertEqual('c8a8c328f655178bdfd600a7710cf8e1',
                         mls.Utils.md5_file(os.path.join(log.directory, 'evaluation.json')))

    def test_load_context(self):
        context = mls.sl.models.Context(eval_type=mls.sl.models.EvaluationSupervised)
        directory = os.path.dirname(__file__)
        log = mls.Logging(os.path.join(directory, '../../files/slw/'), base_dir='')
        context.load(log)
        self.assertIsInstance(context.dataset, mls.sl.datasets.NClassRandomClassificationWithNoise)
        self.assertEqual('NClassRandomClassificationWithNoise', context.dataset.t)
        self.assertIsInstance(context.raw_data, mls.sl.models.Data)
        self.assertEqual(100, len(context.raw_data.x))
        self.assertEqual(100, len(context.raw_data.y))
        self.assertEqual(100, len(context.raw_data.y_pred))
        self.assertIsInstance(context.data, mls.sl.models.Data)
        self.assertEqual(100, len(context.data.x))
        self.assertEqual(100, len(context.data.y))
        self.assertEqual(100, len(context.data.y_pred))
        self.assertIsInstance(context.data_test, mls.sl.models.Data)
        self.assertEqual(20, len(context.data_test.x))
        self.assertEqual(20, len(context.data_test.y))
        self.assertEqual(20, len(context.data_test.y_pred))
        self.assertIsInstance(context.data_train, mls.sl.models.Data)
        self.assertEqual(80, len(context.data_train.x))
        self.assertEqual(80, len(context.data_train.y))
        self.assertEqual(80, len(context.data_train.y_pred))
        self.assertIsInstance(context.algorithm, mls.sl.models.Algorithm)
        self.assertEqual(15, context.algorithm.hyperparameters['n_neighbors'])
        self.assertEqual('sklearn.neighbors.KNeighborsClassifier', context.algorithm.algorithm_family)
        self.assertIsInstance(context.classifier, neighbors.KNeighborsClassifier)
        self.assertIsInstance(context.evaluation, mls.sl.models.EvaluationSupervised)
        self.assertEqual(0.95, context.evaluation.score)
        np.testing.assert_array_equal(np.array([[13, 0], [1, 6]]), context.evaluation.confusion_matrix)

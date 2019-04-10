import os
import shutil
import unittest
import uuid

import sklearn.neighbors as neighbors

import mlsurvey as mls


class TestContext(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        log = mls.Logging()
        shutil.rmtree(log.base_dir)

    def test_init_all_init(self):
        context = mls.models.Context(eval_type=mls.models.Evaluation)
        self.assertIsInstance(context.id, uuid.UUID)
        self.assertIsInstance(context.dataset, mls.datasets.DataSet)
        self.assertIsInstance(context.data, mls.models.Data)
        self.assertIsInstance(context.data_train, mls.models.Data)
        self.assertIsInstance(context.data_test, mls.models.Data)
        self.assertIsNone(context.algorithm)
        self.assertIsNone(context.classifier)
        self.assertIsInstance(context.evaluation, mls.models.Evaluation)

    def test_init_context_supervised(self):
        context = mls.models.Context(eval_type=mls.models.EvaluationSupervised)
        self.assertIsInstance(context.evaluation, mls.models.EvaluationSupervised)

    def test_init_unknown_eval_type(self):
        """
        :test : mlsurvey.models.Context()
        :condition : unknown evaluation
        :main_result : raise AttributeError
        """
        try:
            # This line makes a unresolved reference warning but that's what i want to test the context
            _ = mls.models.Context(eval_type=mls.models.UnknownEvaluation)
            self.assertTrue(False)
        except AttributeError:
            self.assertTrue(True)

    def test_save_context(self):
        context = mls.models.Context(eval_type=mls.models.EvaluationSupervised)
        config_algo = {
            'algorithm-family': 'sklearn.neighbors.KNeighborsClassifier',
            'hyperparameters': {
                'n_neighbors': 3,
                'algorithm': 'auto',
                'weights': 'uniform'
            }
        }
        context.algorithm = mls.models.Algorithm(config_algo)
        log = mls.Logging()
        context.save(log)
        self.assertEqual(len(os.listdir(log.directory)), 5)
        self.assertTrue(os.path.isfile(log.directory + 'dataset.json'))
        self.assertEqual('10f8ce765f59999d2b3b798cc3267845', mls.Utils.md5_file(log.directory + 'dataset.json'))
        self.assertTrue(os.path.isfile(log.directory + 'input.json'))
        self.assertEqual('a504b11fff5b641f340f193dcd641139', mls.Utils.md5_file(log.directory + 'input.json'))
        self.assertTrue(os.path.isfile(log.directory + 'algorithm.json'))
        self.assertEqual('77ead84d71a01a8e83c8706b2b96cf57', mls.Utils.md5_file(log.directory + 'algorithm.json'))
        self.assertTrue(os.path.isfile(log.directory + 'model.joblib'))
        self.assertEqual('4dc6000a33a1ca18a3aaedb7f9802955', mls.Utils.md5_file(log.directory + 'model.joblib'))
        self.assertTrue(os.path.isfile(log.directory + 'evaluation.json'))
        self.assertEqual('6295378c214948496a72d5de81cb8dcc', mls.Utils.md5_file(log.directory + 'evaluation.json'))

    def test_save_context_no_algorithm_neither_classifier_should_save(self):
        context = mls.models.Context(eval_type=mls.models.EvaluationSupervised)
        log = mls.Logging()
        context.save(log)
        self.assertEqual(len(os.listdir(log.directory)), 3)
        self.assertTrue(os.path.isfile(log.directory + 'dataset.json'))
        self.assertEqual('10f8ce765f59999d2b3b798cc3267845', mls.Utils.md5_file(log.directory + 'dataset.json'))
        self.assertTrue(os.path.isfile(log.directory + 'input.json'))
        self.assertEqual('a504b11fff5b641f340f193dcd641139', mls.Utils.md5_file(log.directory + 'input.json'))
        self.assertTrue(os.path.isfile(log.directory + 'evaluation.json'))
        self.assertEqual('6295378c214948496a72d5de81cb8dcc', mls.Utils.md5_file(log.directory + 'evaluation.json'))

    def test_load_context(self):
        context = mls.models.Context(eval_type=mls.models.EvaluationSupervised)
        directory = os.path.dirname(__file__)
        log = mls.Logging(os.path.join(directory, '../files/slw/'), base_dir='')
        context.load(log)
        self.assertIsInstance(context.dataset, mls.datasets.NClassRandomClassificationWithNoise)
        self.assertEqual(context.dataset.t, 'NClassRandomClassificationWithNoise')
        self.assertIsInstance(context.data, mls.models.Data)
        self.assertEqual(100, len(context.data.x))
        self.assertEqual(100, len(context.data.y))
        self.assertIsInstance(context.data_test, mls.models.Data)
        self.assertEqual(20, len(context.data_test.x))
        self.assertEqual(20, len(context.data_test.y))
        self.assertIsInstance(context.data_train, mls.models.Data)
        self.assertEqual(80, len(context.data_train.x))
        self.assertEqual(80, len(context.data_train.y))
        self.assertIsInstance(context.algorithm, mls.models.Algorithm)
        self.assertEqual(context.algorithm.hyperparameters['n_neighbors'], 15)
        self.assertEqual(context.algorithm.algorithm_family, 'knn')
        self.assertIsInstance(context.classifier, neighbors.KNeighborsClassifier)
        self.assertIsInstance(context.evaluation, mls.models.EvaluationSupervised)
        self.assertEqual(0.95, context.evaluation.score)

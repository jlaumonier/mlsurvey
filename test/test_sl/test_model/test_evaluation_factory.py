import unittest

import numpy as np

import mlsurvey as mls


class TestEvaluationFactory(unittest.TestCase):
    fac = {}

    @classmethod
    def setUpClass(cls):
        cls.fac = dict(mls.sl.models.EvaluationFactory.factories)

    @classmethod
    def tearDownClass(cls):
        mls.sl.models.EvaluationFactory.factories = dict(cls.fac)

    def setUp(self):
        mls.sl.models.EvaluationFactory().factories.clear()

    def test_init_evaluation_factory_should_be_initialized(self):
        evaluation_factory = mls.sl.models.EvaluationFactory()
        self.assertIsNotNone(evaluation_factory)
        self.assertDictEqual({}, evaluation_factory.factories)

    def test_add_factory_should_be_added(self):
        mls.sl.models.EvaluationFactory.add_factory('EvaluationSupervised',
                                                    mls.sl.models.EvaluationFactory)
        self.assertEqual(1, len(mls.sl.models.EvaluationFactory.factories))

    def test_create_evaluation_should_be_generated(self):
        evaluation_factory = mls.sl.models.EvaluationFactory()
        mls.sl.models.EvaluationFactory.add_factory('EvaluationSupervised',
                                                    mls.sl.models.EvaluationSupervised.Factory)
        evaluation = evaluation_factory.create_instance('EvaluationSupervised')
        self.assertIsInstance(evaluation, mls.sl.models.EvaluationSupervised)
        self.assertEqual(0.0, evaluation.score)

    def test_create_evaluation_from_dict_created(self):
        expected_cm = np.array([[1, 2], [3, 4]])
        source = {'type': 'EvaluationSupervised', 'score': 0.55, 'confusion_matrix': expected_cm.tolist()}
        evaluation_factory = mls.sl.models.EvaluationFactory()
        mls.sl.models.EvaluationFactory.add_factory('EvaluationSupervised',
                                                    mls.sl.models.EvaluationSupervised.Factory)
        evaluation = evaluation_factory.create_instance_from_dict(source)
        self.assertIsInstance(evaluation, mls.sl.models.EvaluationSupervised)
        self.assertEqual(0.55, evaluation.score)
        np.testing.assert_array_equal(expected_cm, evaluation.confusion_matrix)

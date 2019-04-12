import unittest

import mlsurvey as mls


class TestEvaluationFactory(unittest.TestCase):
    fac = {}

    @classmethod
    def setUpClass(cls):
        cls.fac = dict(mls.models.EvaluationFactory.factories)

    @classmethod
    def tearDownClass(cls):
        mls.models.EvaluationFactory.factories = dict(cls.fac)

    def setUp(self):
        mls.models.EvaluationFactory().factories.clear()

    def test_init_evaluation_factory_should_be_initialized(self):
        evaluation_factory = mls.models.EvaluationFactory()
        self.assertIsNotNone(evaluation_factory)
        self.assertDictEqual({}, evaluation_factory.factories)

    def test_add_factory_should_be_added(self):
        mls.models.EvaluationFactory.add_factory('EvaluationSupervised',
                                                 mls.models.EvaluationFactory)
        self.assertEqual(1, len(mls.models.EvaluationFactory.factories))

    def test_create_evaluation_should_be_generated(self):
        evaluation_factory = mls.models.EvaluationFactory()
        mls.models.EvaluationFactory.add_factory('EvaluationSupervised',
                                                 mls.models.EvaluationSupervised.Factory)
        evaluation = evaluation_factory.create_instance('EvaluationSupervised')
        self.assertIsInstance(evaluation, mls.models.EvaluationSupervised)
        self.assertEqual(0.0, evaluation.score)

    def test_create_evaluation_from_dict_created(self):
        source = {'type': 'EvaluationSupervised', 'score': 0.55}
        evaluation_factory = mls.models.EvaluationFactory()
        mls.models.EvaluationFactory.add_factory('EvaluationSupervised',
                                                 mls.models.EvaluationSupervised.Factory)
        evaluation = evaluation_factory.create_instance_from_dict(source)
        self.assertIsInstance(evaluation, mls.models.EvaluationSupervised)
        self.assertEqual(0.55, evaluation.score)

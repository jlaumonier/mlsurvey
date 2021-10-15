import numpy as np

from .evaluation import Evaluation
from .evaluation_factory import EvaluationFactory


class EvaluationSupervised(Evaluation):

    def __init__(self):
        super().__init__()
        self.score = 0.0
        self.confusion_matrix = np.array([])
        self.support = {}
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.per_label = {}
        self.sub_evaluation = None

    def to_dict(self):
        eval_dict = super().to_dict()
        supervised_dict = {'score': self.score,
                           'accuracy': self.accuracy,
                           'precision': self.precision,
                           'recall': self.recall,
                           'f1': self.f1,
                           'confusion_matrix': self.confusion_matrix.tolist(),
                           'support': self.support,
                           'per_label': self.per_label}
        if self.sub_evaluation is not None:
            supervised_dict['sub_evaluation'] = self.sub_evaluation.to_dict()
        result = {**eval_dict, **supervised_dict}
        return result

    def from_dict(self, d_src):
        self.score = d_src['score']
        self.confusion_matrix = np.array(d_src['confusion_matrix'])
        self.precision = d_src['precision']
        self.accuracy = d_src['accuracy']
        self.recall = d_src['recall']
        self.f1 = d_src['f1']
        self.support = d_src.get('support', {})
        self.per_label = d_src['per_label']
        if 'sub_evaluation' in d_src:
            self.sub_evaluation = EvaluationFactory.create_instance_from_dict(d_src['sub_evaluation'])

    class Factory:
        @staticmethod
        def create(): return EvaluationSupervised()


EvaluationFactory.add_factory('EvaluationSupervised', EvaluationSupervised.Factory)

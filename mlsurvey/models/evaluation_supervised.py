import numpy as np

from .evaluation import Evaluation
from .evaluation_factory import EvaluationFactory


class EvaluationSupervised(Evaluation):

    def __init__(self):
        super().__init__()
        self.score = 0.0
        self.confusion_matrix = np.array([])
        self.sub_evaluation = None

    def to_dict(self):
        eval_dict = super().to_dict()
        supervised_dict = {'score': self.score, 'confusion_matrix': self.confusion_matrix.tolist()}
        if self.sub_evaluation is not None:
            supervised_dict['sub_evaluation'] = self.sub_evaluation.to_dict()
        result = {**eval_dict, **supervised_dict}
        return result

    def from_dict(self, d_src):
        self.score = d_src['score']
        self.confusion_matrix = np.array(d_src['confusion_matrix'])
        if 'sub_evaluation' in d_src:
            self.sub_evaluation = EvaluationFactory.create_instance_from_dict(d_src['sub_evaluation'])

    class Factory:
        @staticmethod
        def create(): return EvaluationSupervised()


EvaluationFactory.add_factory('EvaluationSupervised', EvaluationSupervised.Factory)

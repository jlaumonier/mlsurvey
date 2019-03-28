import numpy as np

from .evaluation import Evaluation
from .evaluation_factory import EvaluationFactory


class EvaluationFairness(Evaluation):

    def __init__(self):
        super().__init__()
        self.probability = np.array([])

    def to_dict(self):
        eval_dict = super().to_dict()
        supervised_dict = {'probability': self.probability.tolist()}
        result = {**eval_dict, **supervised_dict}
        return result

    def from_dict(self, d_src):
        self.probability = np.array(d_src['probability'])

    class Factory:
        @staticmethod
        def create(): return EvaluationFairness()


EvaluationFactory.add_factory('EvaluationFairness', EvaluationFairness.Factory)

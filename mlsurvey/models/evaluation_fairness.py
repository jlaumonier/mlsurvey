import numpy as np

from .evaluation import Evaluation
from .evaluation_factory import EvaluationFactory


class EvaluationFairness(Evaluation):

    def __init__(self):
        super().__init__()
        self.probability = np.array([])
        self.demographic_parity = 0

    def to_dict(self):
        eval_dict = super().to_dict()
        fairness_dict = {'probability': self.probability.tolist(),
                         'demographic_parity': self.demographic_parity}
        result = {**eval_dict, **fairness_dict}
        return result

    def from_dict(self, d_src):
        self.probability = np.array(d_src['probability'])
        self.demographic_parity = d_src['demographic_parity']

    class Factory:
        @staticmethod
        def create(): return EvaluationFairness()


EvaluationFactory.add_factory('EvaluationFairness', EvaluationFairness.Factory)

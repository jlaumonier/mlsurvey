from .evaluation import Evaluation
from .evaluation_factory import EvaluationFactory


class EvaluationSupervised(Evaluation):

    def __init__(self):
        super().__init__()
        self.score = 0.0

    def to_dict(self):
        eval_dict = super().to_dict()
        supervised_dict = {'score': self.score}
        result = {**eval_dict, **supervised_dict}
        return result

    def from_dict(self, d_src):
        self.score = d_src['score']

    class Factory:
        @staticmethod
        def create(): return EvaluationSupervised()


EvaluationFactory.add_factory('EvaluationSupervised', EvaluationSupervised.Factory)

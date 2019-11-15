from .evaluation import Evaluation
from .evaluation_factory import EvaluationFactory


class EvaluationFairness(Evaluation):

    def __init__(self):
        super().__init__()
        self.demographic_parity = 0
        self.equal_opportunity = None
        self.statistical_parity = None
        self.average_equalized_odds = None
        self.disparate_impact_rate = None

    def to_dict(self):
        eval_dict = super().to_dict()
        fairness_dict = {'demographic_parity': self.demographic_parity,
                         'equal_opportunity': self.equal_opportunity,
                         'statistical_parity': self.statistical_parity,
                         'average_equalized_odds': self.average_equalized_odds,
                         'disparate_impact_rate': self.disparate_impact_rate}
        result = {**eval_dict, **fairness_dict}
        return result

    def from_dict(self, d_src):
        self.demographic_parity = d_src['demographic_parity']
        self.equal_opportunity = d_src['equal_opportunity']
        self.statistical_parity = d_src['statistical_parity']
        self.average_equalized_odds = d_src['average_equalized_odds']
        self.disparate_impact_rate = d_src['disparate_impact_rate']

    class Factory:
        @staticmethod
        def create(): return EvaluationFairness()


EvaluationFactory.add_factory('EvaluationFairness', EvaluationFairness.Factory)

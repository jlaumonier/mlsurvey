import operator
from functools import reduce
from operator import eq


class FairnessUtils:

    @classmethod
    def calculate_cond_probability(cls, data, ofs, givens):
        """ calculate conditional probability of 'ofs' given 'givens' """
        givens_conditions = []
        ofs_conditions = []
        for of in ofs:
            ofs_conditions.append(eq(data.df[of[0]], of[1]))
        for given in givens:
            givens_conditions.append(eq(data.df[given[0]], given[1]))
            ofs_conditions.append(eq(data.df[given[0]], given[1]))

        ofs_logical_conditions = reduce(operator.and_, ofs_conditions)
        givens_logical_conditions = reduce(operator.and_, givens_conditions)

        set_ofs_inter_givens = data.df.loc[ofs_logical_conditions]
        card_ofs_inter_givens = len(set_ofs_inter_givens.index)
        set_givens = data.df.loc[givens_logical_conditions]
        card_givens = len(set_givens.index)
        result = card_ofs_inter_givens / card_givens
        return result

import itertools


class ExpandDict:

    @classmethod
    def dict_generator_cartesian_product(cls, source):
        """ get a dictionary containing lists and calculate the cartesian product of these lists.
            return a generator of dictionaries
        """
        keys = []
        vals = []
        for k, v in source.items():
            keys.append(k)
            if isinstance(v, list):
                vals.append(v)
            else:
                vals.append([v])
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    @staticmethod
    def _run_one_and_node(dictionary: dict) -> list:
        return [dictionary]

    @staticmethod
    def _run_one_or_node(dictionary: dict) -> list:
        return list(ExpandDict.dict_generator_cartesian_product(dictionary))

    @staticmethod
    def _run_one_dict(k: str, value: dict) -> list:
        sub_partial_result = []
        sub_results = ExpandDict.run(value)
        for sub_dict in sub_results:
            sub_partial_result.append({k: sub_dict})
        return sub_partial_result

    @classmethod
    def run(cls, dictionary: dict) -> list:
        """
        Calculate the cartesian product of a dictionary containing lists for alternative values
        """
        result = []
        partial_result = []
        for key, value in dictionary.items():
            if isinstance(value, dict):
                partial_result.append(ExpandDict._run_one_dict(key, value))
            elif isinstance(value, list):
                if isinstance(value[0], dict):  # assume that all elements are dict in the list
                    res = []
                    for v in value:
                        res.extend(ExpandDict._run_one_dict(key, v))
                    partial_result.append(res)
                else:
                    partial_result.append(ExpandDict._run_one_or_node({key: value}))
            else:
                partial_result.append(ExpandDict._run_one_and_node({key: value}))
        for element in itertools.product(*partial_result):
            sub_dict = {}
            for e in element:
                sub_dict.update(e)
            result.append(sub_dict)
        return result

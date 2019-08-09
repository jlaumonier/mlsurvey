import numpy as np
import pandas as pd


class FairnessUtils:

    @classmethod
    def calculate_probability(cls, data):
        """ calculate probabity of each class"""
        df = pd.DataFrame(data=data.x)
        df['target'] = data.y
        target_count = df.target.value_counts()
        nb_classes = target_count.size
        nb_examples = df.target.size
        result = np.zeros((nb_classes,))
        for idc, nbc in enumerate(target_count):
            result[idc] = nbc / nb_examples
        return result

    @classmethod
    def calculate_all_cond_probability(cls, data):
        """ calculate probability of each class given each attribute value (only one attribute)"""
        result = []
        df = pd.DataFrame(data=data.x)
        nb_attribute = df.shape[1]
        df['target'] = data.y
        target_uniq_values = np.sort(df['target'].unique())
        for attr_index in range(nb_attribute):
            proba_attrib = {}
            attr_uniq_values = np.sort(df.iloc[:, attr_index].unique())
            grouped_freq_target = df.groupby([attr_index, 'target']).size()
            grouped_freq = df.groupby([attr_index]).size()
            for attr_value in attr_uniq_values:
                proba_attrib_value = []
                for idc in target_uniq_values:
                    if idc in grouped_freq_target[attr_value]:
                        p = grouped_freq_target[attr_value, idc] / grouped_freq[attr_value]
                    else:
                        p = 0.0
                    proba_attrib_value.append(p)
                proba_attrib[attr_value] = proba_attrib_value
            result.append(proba_attrib)
        return result

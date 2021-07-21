import uuid

import mlsurvey as mls


class Context:

    def __init__(self, eval_type):
        self.id = uuid.uuid1()
        self.dataset = mls.sl.datasets.DataSet('generic')
        self.raw_data = None
        self.data = None
        self.data_train = None
        self.data_test = None
        self.algorithm = None
        self.classifier = None
        self.evaluation = eval_type()

    def save(self, log):
        log.save_dict_as_json('dataset.json', self.dataset.to_dict())
        inputs = {'raw_data': self.raw_data, 'data': self.data, 'train': self.data_train, 'test': self.data_test}
        log.save_input(inputs)
        if self.algorithm is not None:
            log.save_dict_as_json('algorithm.json', self.algorithm.to_dict())
            log.save_classifier(self.classifier)
        log.save_dict_as_json('evaluation.json', self.evaluation.to_dict())

    def load(self, log):
        log.set_sub_dir('LoadDataTask')
        dataset_dict = log.load_json_as_dict('dataset.json')
        self.dataset = mls.sl.datasets.DataSetFactory.create_dataset_from_dict(dataset_dict)
        self.raw_data = log.load_input('raw_data.json')['raw_data']
        log.set_sub_dir('PrepareDataTask')
        self.data = log.load_input('data.json')['data']
        log.set_sub_dir('SplitDataTask')
        split_inputs = log.load_input('split_data.json')
        self.data_train = split_inputs['train']
        self.data_test = split_inputs['test']
        log.set_sub_dir('LearnTask')
        algorithm_dict = log.load_json_as_dict('algorithm.json')
        self.algorithm = mls.sl.models.Algorithm.from_dict(algorithm_dict)
        self.classifier = log.load_classifier()
        log.set_sub_dir('EvaluateTask')
        evaluation_dict = log.load_json_as_dict('evaluation.json')
        self.evaluation = mls.sl.models.EvaluationFactory.create_instance_from_dict(evaluation_dict)
        log.set_sub_dir('')

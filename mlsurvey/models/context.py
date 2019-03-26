import uuid

import mlsurvey as mls


class Context:

    def __init__(self):
        self.id = uuid.uuid1()
        self.dataset = mls.datasets.DataSet('generic')
        self.data = mls.models.Data()
        self.data_train = mls.models.Data()
        self.data_test = mls.models.Data()
        self.algorithm = None
        self.classifier = None
        self.score = 0.0

    def save(self, log):
        log.save_dict_as_json('dataset.json', self.dataset.to_dict())
        inputs = {'data': self.data, 'train': self.data_train, 'test': self.data_test}
        log.save_input(inputs)
        log.save_dict_as_json('algorithm.json', self.algorithm.to_dict())
        log.save_classifier(self.classifier)
        evaluation = {'score': self.score}
        log.save_dict_as_json('evaluation.json', evaluation)

    def load(self, log):
        dataset_dict = log.load_json_as_dict('dataset.json')
        self.dataset = mls.datasets.DataSetFactory.create_dataset_from_dict(dataset_dict)
        inputs = log.load_input('input.json')
        self.data = inputs['data']
        self.data_train = inputs['train']
        self.data_test = inputs['test']
        algorithm_dict = log.load_json_as_dict('algorithm.json')
        self.algorithm = mls.models.Algorithm.from_dict(algorithm_dict)
        self.classifier = log.load_classifier()
        evaluation = log.load_json_as_dict('evaluation.json')
        self.score = evaluation['score']

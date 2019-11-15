class Evaluation:

    def __init__(self):
        pass

    def to_dict(self):
        result = {'type': type(self).__qualname__}
        return result

    def from_dict(self, source_dict):
        pass

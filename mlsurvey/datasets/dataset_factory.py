class DataSetFactory:
    factories = {}

    @staticmethod
    def add_factory(name, dataset_factory):
        DataSetFactory.factories[name] = dataset_factory

    @staticmethod
    def create_dataset(name):
        return DataSetFactory.factories[name].create()

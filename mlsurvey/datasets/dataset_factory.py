class DataSetFactory:
    factories = {}

    @staticmethod
    def add_factory(name, dataset_factory):
        DataSetFactory.factories[name] = dataset_factory

    @staticmethod
    def create_dataset(name):
        factory_name = name
        if name not in DataSetFactory.factories.keys():
            factory_name = 'generic'
        return DataSetFactory.factories[factory_name].create(name)

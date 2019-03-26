class DataSetFactory:
    factories = {}

    @staticmethod
    def add_factory(name, dataset_factory):
        """
        Add a DataSetFactory into all the factories
        :param name: the name of the factory
        :param dataset_factory: the instance of the factory
        """
        DataSetFactory.factories[name] = dataset_factory

    @staticmethod
    def create_dataset(name):
        """
        create a dataset from its name (generic, function name in sklearn.dataset or an existing class in mlsurvey)
        :param name: the name of the dataset
        :return: the instance of the dataset
        """
        factory_name = name
        if name not in DataSetFactory.factories.keys():
            factory_name = 'generic'
        return DataSetFactory.factories[factory_name].create(name)

    @staticmethod
    def create_dataset_from_dict(d):
        """
        create a dataset from a dictionary containing its type and parameters (see Dataset)
        :param d: the dictionary
        :return: the instance of the dataset
        """
        factory_name = d['type']
        if d['type'] not in DataSetFactory.factories.keys():
            factory_name = 'generic'
        dataset = DataSetFactory.factories[factory_name].create(d['type'])
        dataset.set_generation_parameters(d['parameters'])
        return dataset

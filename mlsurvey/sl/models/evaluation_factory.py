class EvaluationFactory:
    factories = {}

    @staticmethod
    def add_factory(name, evaluation_factory):
        """
        Add a EvaluationFactory into all the factories
        :param name: the name of the factory
        :param evaluation_factory: the instance of the factory
        """
        EvaluationFactory.factories[name] = evaluation_factory

    @staticmethod
    def create_instance(name):
        """
        create an evaluation instance from its name
        :param name: the name of the evaluation
        :return: the instance of the evaluation
        """
        return EvaluationFactory.factories[name].create()

    @staticmethod
    def create_instance_from_dict(d):
        """
        create an evaluation instance from a dictionary containing its type
        :param d: the dictionary
        :return: the instance of the evaluation
        """
        name = d['type']
        evaluation = EvaluationFactory.factories[name].create()
        evaluation.from_dict(d)
        return evaluation

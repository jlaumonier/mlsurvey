from mlsurvey.rl.common.environment import Environment


class BaseObject:

    def __init__(self, environment: Environment, name: str, parent=None):
        """
        Constructor of the BaseObject
        :param environment the environment
        :param name name of the BaseObject
        """
        self.environment = environment
        self.name = name
        self.parent = parent

    def get_fullname(self):
        """
        return the full name of the base object
        :return the full name
        """
        result = self.name
        if self.parent:
            parent_fullname = self.parent.get_fullname()
            result = parent_fullname + '.' + self.name
        return result


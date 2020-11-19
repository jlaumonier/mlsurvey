

class Action:
    """
    This class represents an action that can be done by an agent.
    """

    ACTION_TYPE_1 = 1
    ACTION_TYPE_2 = 2

    def __init__(self, environment, action_type):
        """
        :param environment the environment where the action is created
        :param action_type type of the action
        """
        self.environment = environment
        self.type = action_type


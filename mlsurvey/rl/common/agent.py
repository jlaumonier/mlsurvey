class Agent:

    def __init__(self, name):
        """
        Constructor of the agent
        :param name name of the agent. Must be unique
        """
        self.name = name
        self.action = None
        self.observation = None

    def choose_action(self):
        """
        choose the next action of the agent
        """
        self.action = 'action3'

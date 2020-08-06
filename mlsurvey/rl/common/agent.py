class Agent:

    def __init__(self):
        """
        Constructor of the agent
        """
        self.action = None
        self.observation = None

    def choose_action(self):
        """
        choose the next action of the agent
        """
        self.action = 'action' + self.observation

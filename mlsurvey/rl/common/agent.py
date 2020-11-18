from mlsurvey.rl.common.environment import Environment
from mlsurvey.rl.common.object import Object


class Agent(Object):

    def __init__(self, environment: Environment, name: str, parent=None):
        super().__init__(environment=environment, name=name, parent=parent)
        self.action = None
        self.observation = None

    def choose_action(self):
        """
        choose the next action of the agent
        """
        self.action = 'action3'

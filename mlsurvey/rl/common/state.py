from mlsurvey.rl.common import BaseObject, Environment
from mlsurvey.rl.common.agent import Agent


class State(BaseObject):

    def __init__(self, environment: Environment, name: str, parent=None):
        super().__init__(environment=environment, name=name, parent=parent)
        self.agents = set()
        self.objects = dict()

    def add_object(self, obj):
        self.objects[obj.name] = obj
        if isinstance(obj, Agent):
            self.agents.add(obj)




from mlsurvey.rl.common import BaseObject, Environment


class Characteristic(BaseObject):

    def __init__(self, environment: Environment, name: str, parent=None):
        super().__init__(environment=environment, name=name, parent=parent)
        self.value = None

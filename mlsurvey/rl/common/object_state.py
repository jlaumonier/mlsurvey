from mlsurvey.rl.common import BaseObject, Environment


class ObjectState(BaseObject):

    def __init__(self, environment: Environment, name: str, parent=None):
        super().__init__(environment=environment, name=name, parent=parent)
        self.name = 'state'
        self.characteristics = dict()
        self.define_default()

    def define_default(self):
        c = self.environment.create_object(name='Step0', bo_type='mlsurvey.rl.common.Characteristic', parent=self)
        c.value = 0
        self.characteristics[c.name] = c

from mlsurvey.rl.common.base_object import BaseObject
from mlsurvey.rl.common.environment import Environment


class Object(BaseObject):

    def __init__(self, environment: Environment, name: str, parent=None):
        super().__init__(environment=environment, name=name, parent=parent)
        self.object_state = self.environment.create_object(name='state',
                                                           bo_type='mlsurvey.rl.common.ObjectState',
                                                           parent=self)

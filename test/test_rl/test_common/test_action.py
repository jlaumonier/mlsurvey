from unittest import TestCase
from mlsurvey.rl.common.action import Action
from mlsurvey.rl.common.environment import Environment


class TestAction(TestCase):

    def test_init(self):
        env = Environment()
        ac = Action(environment=env, action_type=Action.ACTION_TYPE_1)
        self.assertEqual(env, ac.environment)
        self.assertEqual(Action.ACTION_TYPE_1, ac.type)

import unittest

import mlsurvey as mls


class TestState(unittest.TestCase):

    def test_init(self):
        """
        :test : mls.rl.common.State()
        :condition : -
        :main_result: state is initialized
        """
        env = mls.rl.common.Environment()
        state = mls.rl.common.State(environment=env, name='state')
        self.assertIsInstance(state, mls.rl.common.BaseObject)
        self.assertIsInstance(state, mls.rl.common.State)
        self.assertEqual(set(), state.agents)
        self.assertEqual(dict(), state.objects)

    def test_add_object_is_agent(self):
        """
        :test : mls.rl.common.State.add_object()
        :condition : state and agent are created
        :main_result: state contains the agent
        """
        env = mls.rl.common.Environment()
        state = mls.rl.common.State(environment=env, name='state')
        ag = mls.rl.common.Agent(environment=env, name='agent1')
        state.add_object(ag)
        self.assertSetEqual({ag}, state.agents)
        self.assertDictEqual({ag.name: ag}, state.objects)

    def test_add_object(self):
        """
        :test : mls.rl.common.State.add_object()
        :condition : state and object are created
        :main_result: state contains the object
        """
        env = mls.rl.common.Environment()
        state = mls.rl.common.State(environment=env, name='state')
        obj = mls.rl.common.Object(environment=env, name='agent1')
        state.add_object(obj)
        self.assertSetEqual(set(), state.agents)
        self.assertDictEqual({obj.name: obj}, state.objects)

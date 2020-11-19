import unittest

import mlsurvey as mls


class TestAgent(unittest.TestCase):

    def test_init(self):
        """
        :test : mls.rl.common.Agent()
        :condition : -
        :main_result : Agent is initialized
        """
        expected_name = 'AgentName1'
        env = mls.rl.common.Environment()
        ag = mls.rl.common.Agent(environment=env, name=expected_name)
        self.assertIsInstance(ag, mls.rl.common.BaseObject)
        self.assertIsInstance(ag, mls.rl.common.Object)
        self.assertIsInstance(ag, mls.rl.common.Agent)
        self.assertEqual(ag.name, expected_name)
        self.assertIsNone(ag.action)
        self.assertIsNone(ag.observation)

    def test_choose_action(self):
        """
        :test : mls.rl.common.Agent.setObservation()
        :condition: observation is set
        :main_return : the action of the action is set
        """
        env = mls.rl.common.Environment()
        observation = mls.rl.common.State(environment=env, name='state')
        expected_action_type = mls.rl.common.Action.ACTION_TYPE_1
        env = mls.rl.common.Environment()
        ag = mls.rl.common.Agent(environment=env, name='AgentName1')
        ag.observation = observation
        ag.choose_action()
        self.assertEqual(ag.action.type, expected_action_type)

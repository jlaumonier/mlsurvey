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
        ag = mls.rl.common.Agent(name=expected_name)
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
        observation = mls.rl.common.State(id_state=0)
        expected_action = 'action3'
        ag = mls.rl.common.Agent('AgentName1')
        ag.observation = observation
        ag.choose_action()
        self.assertEqual(ag.action, expected_action)

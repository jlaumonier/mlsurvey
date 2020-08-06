import unittest

import mlsurvey as mls


class TestAgent(unittest.TestCase):

    def test_init(self):
        """
        :test : mls.rl.common.Agent()
        :condition : -
        :main_result : Agent is initialized
        """
        ag = mls.rl.common.Agent()
        self.assertIsInstance(ag, mls.rl.common.Agent)
        self.assertIsNone(ag.action)
        self.assertIsNone(ag.observation)

    def test_choose_action(self):
        """
        :test : mls.rl.common.Agent.setObservation()
        :condition: -
        :main_return : the action of the action is set
        """
        observation = '3'
        expected_action = 'action3'
        ag = mls.rl.common.Agent()
        ag.observation = observation
        ag.choose_action()
        self.assertEqual(ag.action, expected_action)

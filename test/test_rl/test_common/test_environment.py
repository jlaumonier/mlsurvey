import unittest

import mlsurvey as mls


class TestEnvironment(unittest.TestCase):

    def test_init(self):
        """
        :test : mls.rl.common.Environment()
        :condition : non limited step environment
        :main_result : Environment is initialized
        """
        env = mls.rl.common.Environment()
        self.assertIsInstance(env, mls.rl.common.Environment)
        self.assertIsInstance(env.agents, dict)
        self.assertDictEqual(env.agents, dict())
        self.assertFalse(env.end_episode)
        self.assertEqual(env.current_step, 0)
        self.assertIsNone(env.game)
        self.assertIsNone(env.current_state)

    def test_getObservationForAgent(self):
        """
        :test : mls.rl.common.Environment.get_observation_for_agent()
        :condition: -
        :main_result: the observation of the agent
        """
        env = mls.rl.common.Environment()
        env.game = mls.rl.common.Game(max_step=10)
        env.current_state = env.game.init_state()
        expected_observation = env.current_state
        real_observation = env.get_observation_for_agent()
        self.assertEqual(expected_observation, real_observation)

    def test_calculate_end_episode_not_end(self):
        """
        :test : mls.rl.common.Environment.calculate_end_episode()
        :condition: no execution of the environment. game is set
        :main_result: episode is not terminated
        """
        env = mls.rl.common.Environment()
        env.game = mls.rl.common.Game(max_step=10)
        env.current_state = env.game.init_state()
        env.calculate_end_episode()
        is_episode_terminated = env.end_episode
        self.assertFalse(is_episode_terminated)

    def test_calculate_next_state(self):
        """
        :test : mls.rl.common.Environment.calculate_next_state()
        :condition : Environment is initialized
        :main_result: one step is pass
        """
        env = mls.rl.common.Environment()
        env.game = mls.rl.common.Game(max_step=10)
        env.current_state = env.game.init_state()
        old_state = env.current_state
        env.calculate_next_state()
        self.assertEqual(env.current_step, 1)
        self.assertIsInstance(env.current_state, mls.rl.common.State)
        self.assertNotEqual(env.current_state, old_state)

    def test_create_agent(self):
        """
        :test : mls.rl.common.Environment.create_agent()
        :condition : Environment is initialized
        :main_result: an agent is created and added into the environment
        """
        env = mls.rl.common.Environment()
        expected_name = 'AgentName1'
        ag = env.create_agent(expected_name)
        self.assertEqual(len(env.agents), 1)
        self.assertEqual(env.agents[expected_name], ag)
        self.assertEqual(ag.name, expected_name)

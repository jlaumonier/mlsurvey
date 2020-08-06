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
        self.assertFalse(env.end_episode)
        self.assertEqual(env.current_step, 0)
        self.assertEqual(env.max_step, -1)

    def test_init_max_step(self):
        """
        :test : mls.rl.common.Environment()
        :condition : initialized max_step
        :main_result : Environment is initialized
        """
        env = mls.rl.common.Environment(max_step=10)
        self.assertIsInstance(env, mls.rl.common.Environment)
        self.assertFalse(env.end_episode)
        self.assertEqual(env.current_step, 0)
        self.assertEqual(env.max_step, 10)

    def test_getObservationForAgent(self):
        """
        :test : mls.rl.common.Environment.getObservationForAgent()
        :condition: -
        :main_result: the observation of the agent
        """
        expected_observation = '0'
        env = mls.rl.common.Environment()
        real_observation = env.get_observation_for_agent()
        self.assertEqual(real_observation, expected_observation)

    def test_calculate_end_episode(self):
        """
        :test : mls.rl.common.Environment.calculateEndEpisode()
        :condition: no execution of the environment
        :main_result: episode is not terminated
        """
        env = mls.rl.common.Environment(max_step=10)
        env.current_step = 10
        env.calculate_end_episode()
        is_episode_terminated = env.end_episode
        self.assertTrue(is_episode_terminated)

    def test_calculate_next_state(self):
        """
        :test : mls.rl.common.Environment.calculateNextState()
        :condition : Environment is initialized
        :main_result: one step is pass
        """
        env = mls.rl.common.Environment()
        env.calculate_next_state()
        self.assertEqual(env.current_step, 1)

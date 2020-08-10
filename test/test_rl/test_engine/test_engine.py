import unittest

import mlsurvey as mls


class TestEngine(unittest.TestCase):

    def test_init(self):
        """
        :test : mls.rl.common.Engine()
        :condition : -
        :main_result : Engine is initialized
        """
        engine = mls.rl.engine.Engine()
        self.assertIsInstance(engine, mls.rl.engine.Engine)
        self.assertIsInstance(engine.environment, mls.rl.common.Environment)

    def test_execute_one_loop(self):
        """
        :test : mls.rl.common.Engine.execute()
        :condition : one agent exists in the environment
        :main_result : 1 loop of the engine
        """
        engine = mls.rl.engine.Engine(max_step=1)
        ag = engine.environment.create_agent(name='agent1')
        old_state = engine.environment.current_state
        engine.execute()
        self.assertEqual(engine.environment.current_step, 1)
        self.assertEqual(ag.action, 'action3')
        self.assertNotEqual(ag.observation, old_state)
        self.assertEqual(ag.observation, engine.environment.current_state)

    def test_execute(self):
        """
        :test : mls.rl.common.Engine.execute()
        :condition : one agent exists in the environment
        :main_result : 10 loop of the engine
        """
        engine = mls.rl.engine.Engine(max_step=10)
        ag = engine.environment.create_agent(name='agent1')
        engine.execute()
        self.assertEqual(engine.environment.current_step, 10)
        self.assertEqual(ag.action, 'action3')
        self.assertEqual(ag.observation, engine.environment.current_state)

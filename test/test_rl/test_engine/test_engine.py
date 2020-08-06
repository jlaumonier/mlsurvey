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
        self.assertIsInstance(engine.agent, mls.rl.common.Agent)

    def test_execute_one_loop(self):
        """
        :test : mls.rl.common.Engine.execute()
        :condition : -
        :main_result : 1 loop of the engine
        """
        engine = mls.rl.engine.Engine(max_step=1)
        engine.execute()
        self.assertEqual(engine.environment.current_step, 1)
        self.assertEqual(engine.agent.action, 'action0')
        self.assertEqual(engine.agent.observation, '1')

    def test_execute(self):
        """
        :test : mls.rl.common.Engine.execute()
        :condition : -
        :main_result : 10 loop of the engine
        """
        engine = mls.rl.engine.Engine(max_step=10)
        engine.execute()
        self.assertEqual(engine.environment.current_step, 10)
        self.assertEqual(engine.agent.action, 'action9')
        self.assertEqual(engine.agent.observation, '10')

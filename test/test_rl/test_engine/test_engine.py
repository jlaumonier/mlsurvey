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
        self.assertIsInstance(engine.environment.game, mls.rl.common.Game)
        self.assertIsInstance(engine.environment.current_state, mls.rl.common.State)

    def test_execute_one_loop(self):
        """
        :test : mls.rl.common.Engine.execute()
        :condition : one agent exists in the environment
        :main_result : 1 loop of the engine
        """
        engine = mls.rl.engine.Engine(max_step=1)
        engine.execute()
        current_state = engine.environment.current_state
        ag = current_state.objects['agent1']
        self.assertEqual(engine.environment.current_step, 1)
        self.assertEqual(ag.action.type, mls.rl.common.Action.ACTION_TYPE_1)
        self.assertEqual(ag.observation, current_state)
        self.assertEqual(1, current_state.objects['object1'].object_state.characteristics['Step0'].value)

    def test_execute(self):
        """
        :test : mls.rl.common.Engine.execute()
        :condition : one agent exists in the environment
        :main_result : 10 loop of the engine
        """
        engine = mls.rl.engine.Engine(max_step=10)
        engine.execute()
        current_state = engine.environment.current_state
        ag = engine.environment.current_state.objects['agent1']
        self.assertEqual(engine.environment.current_step, 10)
        self.assertEqual(ag.action.type, mls.rl.common.Action.ACTION_TYPE_1)
        self.assertEqual(ag.observation, current_state)
        self.assertEqual(10, current_state.objects['object1'].object_state.characteristics['Step0'].value)

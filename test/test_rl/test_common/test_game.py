import unittest

import mlsurvey as mls


class TestGame(unittest.TestCase):

    def test_init(self):
        """
        :test :mls.rl.common.Game()
        :condition : no max steps
        :main_result : Game is initialized
        :return:
        """
        game = mls.rl.common.Game()
        self.assertEqual(game.max_step, None)
        self.assertIsInstance(game, mls.rl.common.Game)

    def test_init_state(self):
        """
        :test : mls.rl.common.Game.init_state()
        :condition : -
        :main_result : the initial state of the game is created
        """
        env = mls.rl.common.Environment()
        game = mls.rl.common.Game()
        state = game.init_state(env)
        agent1 = state.objects['agent1']
        object1 = state.objects['object1']
        # state is a State
        self.assertIsInstance(state, mls.rl.common.State)
        # agent1 is an Agent
        self.assertIsInstance(agent1, mls.rl.common.Agent)
        # object1 is an Object...
        self.assertIsInstance(object1, mls.rl.common.Object)
        # ... but not an agent
        self.assertNotIsInstance(object1, mls.rl.common.Agent)
        # only one agent in state.agents
        self.assertEqual(1, len(state.agents))
        # agent1 is in the agents' state
        self.assertIn(agent1, state.agents)
        # agent1 has state as parent
        self.assertEqual('State.'+agent1.name, agent1.get_fullname())
        # object1 has state as parent
        self.assertEqual('State.'+object1.name, object1.get_fullname())
        # 2 objects in state.objects
        self.assertEqual(2, len(state.objects))
        # object1 is in the objects' state
        self.assertEqual(object1, state.objects[object1.name])
        # object1 is well initialized
        self.assertEqual(0, object1.object_state.characteristics['Step0'].value)
        # 7 objects in environment.objects (State, Agent (ObjectState, Charac), Object (ObjectState, Charac))
        self.assertEqual(7, len(env.objects))

    def test_next_state_state(self):
        """
        :test : mls.rl.common.Game.next_state()
        :condition : current state is a game init state
        :main_result : the next state of the game is calculated
        """
        env = mls.rl.common.Environment()
        game = mls.rl.common.Game()
        current_state = game.init_state(env)
        new_state = game.next_state(current_state)
        self.assertIsInstance(new_state, mls.rl.common.State)
        self.assertEqual(1, new_state.objects['object1'].object_state.characteristics['Step0'].value)

    def test_is_final_state_final(self):
        """
        :test : mls.rl.common.Game.is_final()
        :condition : state is final
        :main_result : the game is in final state
        """
        max_step = 15
        env = mls.rl.common.Environment()
        game = mls.rl.common.Game(max_step=max_step)
        state = mls.rl.common.State(environment=env, name='State')
        state.add_object(env.create_object('object1', 'mlsurvey.rl.common.Object'))
        state.objects['object1'].object_state.characteristics['Step0'].value = max_step
        result = game.is_final(state)
        self.assertTrue(result)

    def test_is_final_state_not_final_with_max_step(self):
        """
        :test : mls.rl.common.Game.is_final()
        :condition : state is not final and max step is set
        :main_result : the game is note in final state
        """
        max_step = 15
        env = mls.rl.common.Environment()
        game = mls.rl.common.Game(max_step=max_step)
        state = mls.rl.common.State(environment=env, name='State')
        state.add_object(env.create_object('object1', 'mlsurvey.rl.common.Object'))
        state.objects['object1'].object_state.characteristics['Step0'].value = 3
        result = game.is_final(state)
        self.assertFalse(result)

    def test_is_final_state_not_final_without_max_step(self):
        """
        :test : mls.rl.common.Game.is_final()
        :condition : state is not final and max step is not set
        :main_result : the game is note in final state
        """
        env = mls.rl.common.Environment()
        game = mls.rl.common.Game()
        state = mls.rl.common.State(environment=env, name='State')
        state.add_object(env.create_object('object1', 'mlsurvey.rl.common.Object'))
        state.objects['object1'].object_state.characteristics['Step0'].value = 3
        result = game.is_final(state)
        self.assertFalse(result)

    def test_observe_state(self):
        """
        :test : mls.rl.common.Game.observe_state()
        :condition : -
        :main_result : the observation is returned as the the entire state
        """
        env = mls.rl.common.Environment()
        game = mls.rl.common.Game()
        state = mls.rl.common.State(environment=env, name='State')
        state.add_object(env.create_object('object1', 'mlsurvey.rl.common.Object'))
        state.objects['object1'].object_state.characteristics['Step0'].value = 3
        result = game.observe_state(state)
        self.assertEqual(result, state)

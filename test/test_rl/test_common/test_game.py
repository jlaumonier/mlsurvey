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
        game = mls.rl.common.Game()
        state = game.init_state()
        self.assertIsInstance(state, mls.rl.common.State)

    def test_next_state_state_id_int(self):
        """
        :test : mls.rl.common.Game.next_state()
        :condition : current state is a state with an int id
        :main_result : the next state of the game is calculated
        """
        game = mls.rl.common.Game()
        current_state = mls.rl.common.State(id_state=0)
        new_state = game.next_state(current_state)
        self.assertIsInstance(new_state, mls.rl.common.State)
        self.assertNotEqual(current_state, new_state)

    def test_is_final_state_final(self):
        """
        :test : mls.rl.common.Game.is_final()
        :condition : state is final
        :main_result : the game is in final state
        """
        max_step = 15
        game = mls.rl.common.Game(max_step=max_step)
        state = mls.rl.common.State(id_state=max_step)
        result = game.is_final(state)
        self.assertTrue(result)

    def test_is_final_state_not_final_with_max_step(self):
        """
        :test : mls.rl.common.Game.is_final()
        :condition : state is not final and max step is set
        :main_result : the game is note in final state
        """
        max_step = 15
        game = mls.rl.common.Game(max_step=max_step)
        state = mls.rl.common.State(id_state=3)
        result = game.is_final(state)
        self.assertFalse(result)

    def test_is_final_state_not_final_without_max_step(self):
        """
        :test : mls.rl.common.Game.is_final()
        :condition : state is not final and max step is not set
        :main_result : the game is note in final state
        """
        game = mls.rl.common.Game()
        state = mls.rl.common.State(id_state=3)
        result = game.is_final(state)
        self.assertFalse(result)

    def test_observe_state(self):
        """
        :test : mls.rl.common.Game.observe_state()
        :condition : -
        :main_result : the observation is returned as the the entire state
        """
        game = mls.rl.common.Game()
        state = mls.rl.common.State(id_state=3)
        result = game.observe_state(state)
        self.assertEqual(result, state)

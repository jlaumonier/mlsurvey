import unittest

import mlsurvey as mls


class TestState(unittest.TestCase):

    def test_init_id_set(self):
        """
        :test : mls.rl.common.State()
        :condition : id is set
        :main_result: state is initialized with the correct id
        """
        state = mls.rl.common.State(id_state=10)
        self.assertIsInstance(state, mls.rl.common.State)
        self.assertEqual(state.id, 10)

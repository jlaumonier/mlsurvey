import unittest
import uuid

import mlsurvey as mls


class TestState(unittest.TestCase):

    def test_init(self):
        """
        :test : mls.rl.common.State()
        :condition : -
        :main_result: state is initialized
        """
        state = mls.rl.common.State()
        self.assertIsInstance(state, mls.rl.common.State)
        self.assertIsInstance(state.id, uuid.UUID)

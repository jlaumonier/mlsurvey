from unittest import TestCase
import mlsurvey as mls


class TestObjectState(TestCase):

    def test_init(self):
        """
        :test : mls.rl.common.ObjectState()
        :condition : -
        :main_result : ObjectState is initialized
        """
        expected_name = 'state'
        env = mls.rl.common.Environment()
        object_state = mls.rl.common.ObjectState(environment=env, name='false_name')
        self.assertIsInstance(object_state, mls.rl.common.BaseObject)
        self.assertIsInstance(object_state, mls.rl.common.ObjectState)
        self.assertEqual(env, object_state.environment)
        self.assertEqual(expected_name, object_state.name)
        self.assertEqual(1, len(object_state.characteristics))
        self.assertEqual('Step0', list(object_state.characteristics.keys())[0])
        charac = object_state.characteristics['Step0']
        self.assertEqual(0, charac.value)
        self.assertEqual('state.Step0', charac.get_fullname())
        self.assertIn(charac.get_fullname(), env.objects)

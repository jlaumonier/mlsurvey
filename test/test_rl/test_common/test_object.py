from unittest import TestCase
import mlsurvey as mls


class TestObject(TestCase):

    def test_init(self):
        """
        :test : mls.rl.common.Object()
        :condition : -
        :main_result : Object is initialized
        """
        expected_name = 'Object1'
        env = mls.rl.common.Environment()
        obj = mls.rl.common.Object(environment=env, name=expected_name)
        self.assertIsInstance(obj, mls.rl.common.BaseObject)
        self.assertIsInstance(obj, mls.rl.common.Object)
        self.assertEqual(env, obj.environment)
        self.assertEqual(expected_name, obj.name)
        self.assertIsNotNone(obj.object_state)
        self.assertIsInstance(obj.object_state, mls.rl.common.ObjectState)
        self.assertEqual(expected_name+'.state', obj.object_state.get_fullname())
        self.assertIsNotNone(env.objects[obj.object_state.get_fullname()])

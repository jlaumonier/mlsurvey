from unittest import TestCase

import mlsurvey as mls


class TestCharacteristic(TestCase):

    def test_init(self):
        """
        :test : mls.rl.common.Characteristic()
        :condition : -
        :main_result : Characteristic is initialized
        """
        expected_name = 'Charac1'
        env = mls.rl.common.Environment()
        characteristic = mls.rl.common.Characteristic(environment=env, name=expected_name)
        self.assertIsInstance(characteristic, mls.rl.common.BaseObject)
        self.assertIsInstance(characteristic, mls.rl.common.Characteristic)
        self.assertEqual(env, characteristic.environment)
        self.assertEqual(expected_name, characteristic.name)
        self.assertIsNone(characteristic.value)

    def test_value(self):
        env = mls.rl.common.Environment()
        characteristic1 = mls.rl.common.Characteristic(environment=env, name='Charact1')
        characteristic2 = mls.rl.common.Characteristic(environment=env, name='Charact2')
        characteristic1.value = 'test'
        characteristic2.value = 1
        self.assertIsInstance(characteristic1.value, str)
        self.assertIsInstance(characteristic2.value, int)
        characteristic2.value = 'test2'
        self.assertIsInstance(characteristic2.value, str)

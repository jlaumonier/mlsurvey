from unittest import TestCase

import mlsurvey as mls


class TestBaseObject(TestCase):

    def test_init(self):
        """
        :test : mls.rl.common.BaseObject()
        :condition : -
        :main_result : BaseObject is initialized with no parent
        """
        expected_name = 'BaseObject1'
        env = mls.rl.common.Environment()
        bo = mls.rl.common.BaseObject(environment=env, name=expected_name)
        self.assertIsInstance(bo, mls.rl.common.BaseObject)
        self.assertEqual(env, bo.environment)
        self.assertEqual(expected_name, bo.name)
        self.assertIsNone(bo.parent)

    def test_init_with_parent(self):
        """
        :test : mls.rl.common.BaseObject()
        :condition : parent is created
        :main_result : BaseObject is initialized with parent
        """
        expected_name = 'BaseObject1'
        env = mls.rl.common.Environment()
        bo_parent = mls.rl.common.BaseObject(environment=env, name='Parent1')
        bo = mls.rl.common.BaseObject(environment=env, name=expected_name, parent=bo_parent)
        self.assertEqual(bo_parent, bo.parent)

    def test_get_full_name_no_parent(self):
        """
        :test : mls.rl.common.BaseObject.getFullName()
        :condition : No parent
        :main_result : full name is the same as the name
        """
        expected_name = 'BaseObject1'
        env = mls.rl.common.Environment()
        bo = mls.rl.common.BaseObject(environment=env, name=expected_name)
        self.assertEqual(expected_name, bo.get_fullname())

    def test_get_full_name_with_parent(self):
        """
        :test : mls.rl.common.BaseObject.getFullName()
        :condition : baseobject has a parent
        :main_result : full name is the same as the name + parent name
        """
        expected_name = 'BaseObject1'
        env = mls.rl.common.Environment()
        bo_parent = mls.rl.common.BaseObject(environment=env, name='Parent1')
        bo = mls.rl.common.BaseObject(environment=env, name=expected_name, parent=bo_parent)
        self.assertEqual(bo_parent.get_fullname() + '.' + expected_name, bo.get_fullname())

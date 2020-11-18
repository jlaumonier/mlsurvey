import unittest

import mlsurvey as mls


class TestEnvironment(unittest.TestCase):

    def test_init(self):
        """
        :test : mls.rl.common.Environment()
        :condition : non limited step environment
        :main_result : Environment is initialized
        """
        env = mls.rl.common.Environment()
        self.assertIsInstance(env, mls.rl.common.Environment)
        self.assertIsInstance(env.agents, set)
        self.assertSetEqual(env.agents, set())
        self.assertIsInstance(env.objects, dict)
        self.assertDictEqual(env.objects, dict())
        self.assertFalse(env.end_episode)
        self.assertEqual(env.current_step, 0)
        self.assertIsNone(env.game)
        self.assertIsNone(env.current_state)

    def test_getObservationForAgent(self):
        """
        :test : mls.rl.common.Environment.get_observation_for_agent()
        :condition: -
        :main_result: the observation of the agent
        """
        env = mls.rl.common.Environment()
        env.game = mls.rl.common.Game(max_step=10)
        env.current_state = env.game.init_state(env)
        expected_observation = env.current_state
        real_observation = env.get_observation_for_agent()
        self.assertEqual(expected_observation, real_observation)

    def test_calculate_end_episode_not_end(self):
        """
        :test : mls.rl.common.Environment.calculate_end_episode()
        :condition: no execution of the environment. game is set
        :main_result: episode is not terminated
        """
        env = mls.rl.common.Environment()
        env.game = mls.rl.common.Game(max_step=10)
        env.current_state = env.game.init_state(env)
        env.calculate_end_episode()
        is_episode_terminated = env.end_episode
        self.assertFalse(is_episode_terminated)

    def test_calculate_next_state(self):
        """
        :test : mls.rl.common.Environment.calculate_next_state()
        :condition : Environment is initialized
        :main_result: one step is passed
        """
        env = mls.rl.common.Environment()
        env.game = mls.rl.common.Game(max_step=10)
        env.current_state = env.game.init_state(env)
        env.calculate_next_state()
        self.assertEqual(env.current_step, 1)
        self.assertIsInstance(env.current_state, mls.rl.common.State)
        self.assertEqual(2, len(env.current_state.objects))
        self.assertEqual(1, len(env.current_state.agents))
        self.assertEqual(1, env.current_state.objects['object1'].object_state.characteristics['Step0'].value)

    def test_create_base_object_base_object_created(self):
        """
        :test : mls.rl.common.Environment.create_base_object()
        :condition : Environment is initialized
        :main_result: based objects are created
        """
        env = mls.rl.common.Environment()
        obj = env._create_base_object('object1', 'mlsurvey.rl.common.Object')
        self.assertIsInstance(obj, mls.rl.common.Object)
        self.assertIsNone(obj.parent)

    def test_create_base_object_base_object_created_with_parent(self):
        """
        :test : mls.rl.common.Environment.create_base_object()
        :condition : Environment is initialized, parent exists
        :main_result: based objects are created with parent
        """
        env = mls.rl.common.Environment()
        obj_parent = env._create_base_object('object2', 'mlsurvey.rl.common.Object')
        obj = env._create_base_object(name='object1',
                                      bo_type='mlsurvey.rl.common.Object',
                                      parent=obj_parent)
        self.assertIsInstance(obj, mls.rl.common.Object)
        self.assertEqual(obj_parent, obj.parent)

    def test_create_agent(self):
        """
        :test : mls.rl.common.Environment.create_agent()
        :condition : Environment is initialized. A parent is created
        :main_result: an agent is created and added into the environment
        """
        env = mls.rl.common.Environment()
        expected_name = 'AgentName1'
        obj_parent = env._create_base_object('parent', 'mlsurvey.rl.common.BaseObject')
        ag = env.create_object(expected_name, 'mlsurvey.rl.common.Agent', obj_parent)
        self.assertEqual(len(env.agents), 1)
        self.assertEqual(len(env.objects), 3)
        self.assertEqual(list(env.agents)[0], ag)
        self.assertEqual('parent.AgentName1', ag.get_fullname())
        self.assertEqual(env.objects[ag.get_fullname()], ag)
        self.assertEqual(ag.name, expected_name)

    def test_create_object(self):
        """
        :test : mls.rl.common.Environment.create_object()
        :condition : Environment is initialized
        :main_result: an object is created and added into the environment
        """
        env = mls.rl.common.Environment()
        expected_name = 'ObjectName1'
        obj = env.create_object(expected_name, 'mlsurvey.rl.common.Object')
        self.assertEqual(len(env.objects), 3)
        self.assertEqual(len(env.agents), 0)
        self.assertEqual(env.objects[expected_name], obj)
        self.assertIsInstance(obj, mls.rl.common.Object)
        self.assertEqual(obj.name, expected_name)

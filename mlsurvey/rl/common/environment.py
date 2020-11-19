import mlsurvey as mls


class Environment:

    def __init__(self):
        """
        initialisation of the environment
        """
        self.end_episode = False
        self.current_step = 0
        self.current_state = None
        self.agents = set()
        self.objects = dict()
        self.game = None
        self.next_action = dict()

    def get_observation_for_agent(self):
        """
        get the current observation. current state at the moment
        :return: the current observation
        """
        return self.game.observe_state(self.current_state)

    def calculate_end_episode(self):
        """
        calculate the end of the episode. modify self.end_episode if the conditions of the end of episode are met
        """
        self.end_episode = self.game.is_final(self.current_state)

    def calculate_next_state(self):
        """
        Calculate the next state of the environment.
        Change the current state.
        Increase the step
        """
        self.current_step = self.current_step + 1
        self.current_state = self.game.next_state(current_state=self.current_state, actions=self.next_action)

    def _create_base_object(self, name: str, bo_type: str, parent=None):
        """
        create a base object from a string represent the type
        :param name the identification name of the BaseObject
        :param bo_type the string of the class to create the object
        :param parent the parent of the current object. May be None
        :return: the BaseObject instance
        """
        class_ = mls.Utils.import_from_dotted_path(bo_type)
        result = class_(environment=self, name=name, parent=parent)
        return result

    def create_object(self, name: str, bo_type: str, parent=None):
        """
        Create an object. Add it into the object dictionary and agent set if the object is Agent
        :param name the identification name of the object
        :param bo_type the string of the class to create the object
        :param parent the parent of the current object. May be None
        :return: the object instance
        """
        result = self._create_base_object(name=name, bo_type=bo_type, parent=parent)
        self.objects[result.get_fullname()] = result
        if isinstance(result, mls.rl.common.Agent):
            self.agents.add(result)
        return result

    def create_action(self, action_type):
        """
        create an action
        :param action_type the type of the action (Constants in Action class)
        """
        result = mls.rl.common.Action(environment=self, action_type=action_type)
        return result

    def choose_action(self):
        """
        tell all agent to choose their action
        """
        for ag in self.agents:
            ag.choose_action()
            self.next_action[ag.name] = ag.action

import mlsurvey as mls


class Environment:

    def __init__(self, max_step=-1):
        """
        initialisation of the environment
        :param max_step
        """
        self.end_episode = False
        self.current_step = 0
        self.current_state = mls.rl.common.State()
        self.max_step = max_step
        self.agents = dict()

    def get_observation_for_agent(self):
        """
        get the current observation. current state at the moment
        :return: the current observation
        """
        return self.current_state

    def calculate_end_episode(self):
        """
        calculate the end of the episode. modify self.end_episode if the conditions of the end of episode are met
        """
        if self.current_step == self.max_step:
            self.end_episode = True

    def calculate_next_state(self):
        """
        Calculate the next state of the environment.
        Change the current state.
        Increase the step
        """
        self.current_step = self.current_step + 1
        self.current_state = mls.rl.common.State()

    def create_agent(self, name):
        """
        Create an agent. Add it into the agent set
        :param name the identification name of the agent
        :return: the agent instance
        """
        result = mls.rl.common.Agent(name=name)
        self.agents[name] = result
        return result

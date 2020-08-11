import mlsurvey as mls


class Environment:

    def __init__(self):
        """
        initialisation of the environment
        """
        self.end_episode = False
        self.current_step = 0
        self.current_state = None
        self.agents = dict()
        self.game = None

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
        self.current_state = self.game.next_state(current_state=self.current_state)

    def create_agent(self, name):
        """
        Create an agent. Add it into the agent set
        :param name the identification name of the agent
        :return: the agent instance
        """
        result = mls.rl.common.Agent(name=name)
        self.agents[name] = result
        return result

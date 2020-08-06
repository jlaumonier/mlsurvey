class Environment:

    def __init__(self, max_step=-1):
        """
        initialisation of the environment
        :param max_step
        """
        self.end_episode = False
        self.current_step = 0
        self.max_step = max_step

    def get_observation_for_agent(self):
        """
        get the current observation
        :return: the current observation
        """
        return str(self.current_step)

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


import mlsurvey as mls


class Engine:

    def __init__(self, max_step=-1):
        """
        initialization of the engine
        :param max_step maximum step of the environment. temporary. will be change when adding Game
        """
        self.environment = mls.rl.common.Environment(max_step=max_step)
        self.agent = mls.rl.common.Agent()

    def execute(self):
        """
        Main loop of the application.
        """
        observation = self.environment.get_observation_for_agent()
        self.agent.observation = observation
        while not self.environment.end_episode:
            self.agent.choose_action()
            self.environment.calculate_next_state()
            self.environment.calculate_end_episode()
            observation = self.environment.get_observation_for_agent()
            self.agent.observation = observation

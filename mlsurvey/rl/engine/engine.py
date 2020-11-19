import mlsurvey as mls


class Engine:

    def __init__(self, max_step=-1):
        """
        initialization of the engine
        :param max_step maximum step of the environment. temporary. will be change when adding Game
        """
        self.environment = mls.rl.common.Environment()
        self.environment.game = mls.rl.common.Game(max_step=max_step)
        self.environment.current_state = self.environment.game.init_state(self.environment)

    def execute(self):
        """
        Main loop of the application.
        """
        # set observations for all agents
        observation = self.environment.get_observation_for_agent()
        for ag in self.environment.agents:
            ag.observation = observation
        # main loop
        while not self.environment.end_episode:
            # each agent choose its action
            self.environment.choose_action()
            # next state
            self.environment.calculate_next_state()
            # is the end of the episode
            self.environment.calculate_end_episode()
            # set observations for all agents
            observation = self.environment.get_observation_for_agent()
            for ag in self.environment.agents:
                ag.observation = observation

import mlsurvey as mls


class Game:
    """
    Represents the rules of the game
    """

    def __init__(self, max_step=None):
        """
        initialize the game
        :param max_step: maximum step of the game. None or default for no max step
        """
        self.max_step = max_step
        self._init_id_state = 0
        self._evolving_id_state = 1
        
    def init_state(self):
        """
        Create the initial state of the game
        :return: the initial state
        """
        return mls.rl.common.State(id_state=self._init_id_state)

    def next_state(self, current_state):
        """
        Calculates the new state according to the current state and the action of the agent
        :param current_state the current state
        :return: the new state
        """
        return mls.rl.common.State(id_state=current_state.id + self._evolving_id_state)

    def is_final(self, state):
        """
        Is the state final ?
        :param state: the state
        :return: True if final, False if not
        """
        return state.id == self.max_step

    @staticmethod
    def observe_state(state):
        """
        State Observation function. calculate the partial state according to d from state pS
        :param state the state of the environment
        :return the observed state. The result is a state (possibly partial and noisy) and not a general observation.
        """
        observation = state
        return observation

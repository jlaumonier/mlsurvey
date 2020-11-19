from mlsurvey.rl.common import Environment, State


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

    @staticmethod
    def init_state(env: Environment) -> State:
        """
        Create the initial state of the game
        :return: the initial state
        """
        result = env.create_object(name='State', bo_type='mlsurvey.rl.common.State', parent=None)
        ag = env.create_object(name='agent1',
                               bo_type='mlsurvey.rl.common.Agent',
                               parent=result)
        result.add_object(ag)
        obj = env.create_object(name='object1',
                                bo_type='mlsurvey.rl.common.Object',
                                parent=result)
        result.add_object(obj)
        return result

    @staticmethod
    def next_state(current_state, actions):
        """
        Calculates the new state according to the current state and the action of the agent.
        :param current_state the current state
        :param actions action of the agents
        :return: the new state
        """
        result = current_state
        for ag in current_state.agents:
            action = actions[ag.name]
            result.objects['object1'].object_state.characteristics['Step0'].value += action.type
        return result

    def is_final(self, state):
        """
        Is the state final ?
        :param state: the state
        :return: True if final, False if not
        """
        return state.objects['object1'].object_state.characteristics['Step0'].value == self.max_step

    @staticmethod
    def observe_state(state):
        """
        State Observation function. calculate the partial state according to d from state pS
        :param state the state of the environment
        :return the observed state. The result is a state (possibly partial and noisy) and not a general observation.
        """
        observation = state
        return observation

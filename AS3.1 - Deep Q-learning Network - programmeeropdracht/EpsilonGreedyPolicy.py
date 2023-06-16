import numpy as np
import tensorflow as tf


class EpsilonGreedyPolicy:
    """Epsilon greedy policy class."""

    def __init__(self, env: object, epsilon: float, decay_factor: float, minimal_epsilon: float):
        """Initialize epsilon greedy policy.

        :param env: The environment
        :param epsilon: Base epsilon
        :param decay_factor: Epsilon decay factor
        :param minimal_epsilon: Minimal epsilon
        """

        self.env = env
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.minimal_epsilon = minimal_epsilon

    def select_action(self, state: object, model: object):
        """Select the next action based on the state and policy.

        :param state: Observation of environment
        :param model: Neural network model
        :return: Chosen action
        """

        if np.random.rand(1)[0] > self.epsilon:
            action = np.argmax(model.predict(np.expand_dims(state, axis=0)))
            return action
        else:
            return self.env.action_space.sample()  # Chooses a random action from all possible actions

    def epsilon_decay(self):
        """Add decay to epsilon overtime function optional."""

        if self.epsilon > self.minimal_epsilon:
            self.epsilon = self.epsilon * self.decay_factor

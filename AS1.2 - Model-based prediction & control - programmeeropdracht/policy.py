import abc
import numpy as np
from agent import Action
from typing import List


class Policy(metaclass=abc.ABCMeta):
    """Generic agent class with fill-in template."""

    @abc.abstractmethod
    def decide_action(self, observation: List):
        """Take an action based on the given observation."""
        raise NotImplementedError


class PureRandomPolicy(Policy):
    """A policy that takes action on a purely random basis."""

    def __init__(self):
        """Initialize random Policy."""
        self.visual = False
        self.visual_matrix = None

    def decide_action(self, observation):
        """
        Decide action based on pure randomness.

        :param observation: dict containing information about the environment. NOT USED FOR THIS POLICY
        :return: Random action chosen by policy
        """
        return np.random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])


class ValueBasedPolicy(Policy):
    """A policy that takes action based on value."""

    def __init__(self, gamma=1, visuals=True):
        """
        Create Value-based policy with parameters.

        :param gamma: Gamma is the discount value in this context
        :param visuals: This parameter is used to check if we want to visualize
        """
        self.value_matrix = None
        self.visual_matrix = None
        self.agent = None
        self.gamma = gamma
        self.visual = visuals

    def decide_action(self, observation):
        """
        Decide action based on pure random.

        :param observation: observation is a dict containing information about the environment
        :return: Action chosen based on the observation
        """
        all_action = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        outcome = []
        # Get all values from every action possible
        for action in all_action:
            self.agent.env.reset(observation["agent_location"])
            obs, r, _, _ = self.agent.env.step(action)
            outcome.append((action, r, obs))

        # Return best value using the Bellman equation
        return max(outcome, key=lambda x: x[1] + self.gamma * self.value_matrix[x[2]["agent_location"]])[0]

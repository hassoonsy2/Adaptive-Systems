import numpy as np


class Agent:
    """Agent class"""

    def __init__(self, policy, env=None):
        """Initialize agent with values."""
        self.policy = policy
        self.env = env

    def value_iteration(self, itera: int = 200, gamma: int = 1):
        """
        Calculate values for value function and write to policy.

        :param itera: The amount of maximum iterations used for value iterations.
        :param gamma: This is the discount value used for value iteration
        """
        self.policy.agent = self
        self.policy.gamma = gamma
        self.policy.value_matrix = self.env.maze.copy()

        for i in range(itera):
            new_value_matrix = self.policy.value_matrix.copy()

            for index_y, row in enumerate(self.policy.value_matrix):
                for index_x, _ in enumerate(row):
                    if (index_x, index_y) not in self.env.end_coord:
                        state = (index_y, index_x)
                        action = self.policy.decide_action({"agent_location": state})
                        self.env.reset(state)
                        obs, r, _, _ = self.env.step(action)
                        new_value_matrix[state] = r + gamma * self.policy.value_matrix[obs["agent_location"]]

            if np.allclose(self.policy.value_matrix, new_value_matrix):
                print("No difference from previous iteration")
                break

            self.policy.value_matrix = new_value_matrix

            if self.policy.visual:
                self.visual_function()
                print(f"-----iteration {i}\n{self.policy.value_matrix}\n------\n{self.policy.visual_matrix}\n-----\n")

    def visual_function(self):
        """This function translates the value_matrix to visual matrix."""
        self.policy.visual_matrix = np.zeros((4, 4), dtype=str)
        action_to_string_dict = {0: 'up', 1: 'right', 2: 'down', 3: 'left', 4: 'stay', None: 'None'}

        for index_y, row in enumerate(self.policy.value_matrix):
            for index_x, _ in enumerate(row):
                if (index_x, index_y) not in self.env.end_coord:
                    state = (index_y, index_x)
                    self.env.reset(state)
                    action = self.policy.decide_action({"agent_location": state})
                    self.policy.visual_matrix[state] = action_to_string_dict[action]

    def get_action_from_policy(self, observation):
        """
        Get action from policy.

        :param observation: observation of the world given as dict
        :return: Action decided by the policy
        """
        return self.policy.decide_action(observation)

    def __str__(self):
        return f"{self.policy=}\n"


class Action(int):
    """All possible action available as datatype."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    #STAY = 4

    @staticmethod
    def to_coord_delta(action):
        action_to_coord_delta = {
            Action.UP: (-1, 0),
            Action.RIGHT: (0, 1),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            #Action.STAY: (0, 0)
        }
        return action_to_coord_delta[action]

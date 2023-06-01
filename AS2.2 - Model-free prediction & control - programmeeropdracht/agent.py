
import copy
import numpy as np


class Agent:
    """Agent class itself."""

    def __init__(self, policy, env=None):
        """Initialize agent with values."""
        self.policy = policy
        self.env = env

    def interpret_world(self):
        """Process the world."""
        pass

    def value_iteration(self, itera: int = 200, gamma: int = 1):
        """
        Calculate values for value function and write to policy.

        :param itera: The amount of maximum iterations used for value iterations.
        :param gamma: This is the discount value used for value iteration
        """
        # Write environment to policy because of model based policy
        self.policy.agent = self
        self.policy.gamma = gamma
        self.policy.value_matrix = copy.copy(self.env.maze)

        # Iterate over values
        for i in range(itera):
            # Copy shape
            new_value_matrix = copy.deepcopy(self.policy.value_matrix)

            # Iterate over y and x
            for index_y, x in enumerate(self.policy.value_matrix):
                for index_x, _ in enumerate(x):
                    # Check if it is a terminal state
                    if (index_x, index_y) not in self.env.end_coord:
                        # Set the state that going to be used which is only agent position
                        state = (index_y, index_x)

                        # Let the value based policy decide the most greedy action
                        action = self.policy.decide_action({"agent_location": state})

                        # Reset environment to current state
                        self.env.reset(state)

                        # Use best available action and use Bellman equation
                        obs, r, _, _ = self.env.step(action)
                        new_value_matrix[state] = r + gamma * self.policy.value_matrix[obs["agent_location"]]
            # Check if there is a difference from previous iteration
            if np.allclose(self.policy.value_matrix, new_value_matrix):
                print("No difference from previous iteration")
                break

            # Update matrix
            self.policy.value_matrix = new_value_matrix

            # Visualise the update process
            if self.policy.visual:
                self.visual_function()

                print(f"-----iteration {i}\n{self.policy.value_matrix}\n------\n{self.policy.visual_matrix}\n-----\n")

    def visual_function(self):
        """Translate the value_matrix to visual matrix."""
        self.policy.visual_matrix = np.zeros((4, 4), dtype=str)
        action_to_string_dict = {0: 'up',
                                 1: 'right',
                                 2: 'down',
                                 3: 'left',
                                 4: 'stay',
                                 None: 'None'}

        # Get all values from every action possible
        for index_y, x in enumerate(self.policy.value_matrix):
            for index_x, _ in enumerate(x):
                # Check if it is a terminal state
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

    def save_value_matrix(self, path='text.txt'):
        """Save value matrix to file."""
        np.savetxt(path, self.policy.value_matrix)

    def load_value_matrix(self, path='text.txt'):
        """Load value matrix from file."""
        self.policy.value_matrix = np.genfromtxt(path)

    def __str__(self):
        """
        Debug string return function.

        :return: string of object
        """
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
        """ die een actie neemt en een overeenkomstige coördinatieve delta retourneert
    (dit geeft aan hoe de coördinaten van de agent veranderen als gevolg van het ondernemen van die actie)."""
        action_to_coord_delta = {
            Action.UP: (-1, 0),
            Action.RIGHT: (0, 1),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            #Action.STAY: (0, 0)
        }
        return action_to_coord_delta[action]

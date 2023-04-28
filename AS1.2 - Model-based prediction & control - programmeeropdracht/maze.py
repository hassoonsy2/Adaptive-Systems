import numpy as np
from visu.enviroment_render import render_background, render_in_step
import copy
from agent import Action

class Maze:
    """Class file for the maze as environment."""

    def __init__(self, agent, start_coord=(3, 3), end_coords=[(0, 3), (3, 0)], visualize=False, done=False):
        """
        Create maze with initial values.

        :param agent: Agent used for this environment
        :param start_coord: Start coordinate for the agent
        :param end_coords: Ending coordinates for the maze, terminal states
        :param visualize: Option to visualize the maze
        :param done: Value of the maze to check if the simulation is done.
        """
        self.maze = np.zeros((4, 4))
        self.reward_map = np.array([[-1, -1, -1, 40],
                                    [-1, -1, -10, -10],
                                    [-1, -1, -1, -1],
                                    [10, -2, -1, -1]])
        self.agent = agent
        self.start_coord = start_coord
        self.end_coord = end_coords
        self.agent_location = start_coord
        self.done = done
        self.sim_step = 0
        self.last_action_agent = None
        self.visualize = visualize

        self.rendered_background = render_background(self) if visualize else None
        self.total_reward = 0

        # Give agent copy of env for value function
        self.agent.env = copy.copy(self)

    def step(self, action):
        """
        Step function used for playing out decided actions.

        :param action: Action used in the step
        :return: Returns the observation of the world, the reward from the action, value if done, additional info
        """
        self.sim_step += 1
        self.last_action_agent = action
        action_coord_delta_y, action_coord_delta_x = Action.to_coord_delta(action)

        next_y = self.agent_location[0] + action_coord_delta_y
        next_x = self.agent_location[1] + action_coord_delta_x

        if 0 <= next_y < self.maze.shape[1] and 0 <= next_x < self.maze.shape[0]:
            self.agent_location = (next_y, next_x)

        reward = self.reward_map[self.agent_location]
        self.total_reward += reward

        if self.agent_location in self.end_coord:
            self.done = True

        observation = self.get_state()
        return observation, reward, self.done, {}

    def get_state(self):
        """State function for returning the world as a state to a policy."""
        return {"agent_location": self.agent_location}

    def render(self):
        """Render function for visualizing the maze."""
        return render_in_step(self)

    def reset(self, agent_location):
        """
        Reset env to specific state.

        :param agent_location: The agent location you want to reset to
        """
        self.agent_location = agent_location

    def simulate_step(self, agent_location, action):
        """
        Simulate the step for the given agent location and action.

        :param agent_location: The agent location you want to simulate
        :param action: The action you want to simulate
        :return: Returns the simulated observation, reward, done, and additional info
        """
        next_y = agent_location[0] + Action.to_coord_delta(action)[0]
        next_x = agent_location[1] + Action.to_coord_delta(action)[1]
        next_location = (next_y, next_x)

        if 0 <= next_y < self.maze.shape[1] and 0 <= next_x < self.maze.shape[0]:
            reward = self.reward_map[next_location]
        else:
            next_location = agent_location
            reward = self.reward_map[agent_location]

        done = next_location in self.end_coord

        observation = {"agent_location": next_location}
        return observation, reward, done, {}

    def __str__(self):
        """Return for debugging."""
        return f"{self.maze=}\n" \
               f"{self.reward_map=}\n" \
               f"{self.agent=}\n" \
               f"{self.start_coord=}\n" \
               f"{self.end_coord=}"


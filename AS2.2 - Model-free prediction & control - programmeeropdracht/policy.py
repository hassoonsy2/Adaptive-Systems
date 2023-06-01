import abc
import numpy as np
from agent import Action
from typing import List
import cv2

from PIL import Image, ImageDraw

from pathlib import Path
textures_path = Path(__file__) / '..' / 'visu' / 'textures'

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


class EpsilonSoftGreedyPolicy(Policy):
    """Epsilon Soft greedy policy."""

    def __init__(self, epsilon=0.9):
        """
        Create Epsilon Soft greedy policy with parameters.

        :param epsilon: epsilon used in algorithm.
        """
        self.value_matrix = None
        self.epsilon = epsilon
        self.q_table = None

    def decide_action(self, observation):
        """
        Decide action with highest value in q-table with a Epsilon change to take a random action.

        :param observation: observation is a dict containing information about the environment
        :return: Action chosen based on the observation
        """
        all_actions = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]

        if np.random.rand(1)[0] < self.epsilon:
            agent_pos = observation["agent_location"]
            max_value = max(self.q_table[agent_pos[0]][agent_pos[1]])
            index_action = self.q_table[agent_pos[0]][agent_pos[1]].index(max_value)
            chosen_action = index_action
            return chosen_action
        else:
            return np.random.choice(all_actions)

    def visualise_q_table(self):
        """Visualise q table to img."""
        triangles = Image.open(textures_path / "baground_tiles_for_q.png")

        tile_size_width, tile_size_height = triangles.size

        maze = np.ndarray((4, 4))

        width = maze.shape[0] * tile_size_width
        height = maze.shape[1] * tile_size_height
        background = Image.new(mode="RGB", size=(width, height), color=(70, 0, 255))

        for height_row, width_values in enumerate(maze):
            for index, value in enumerate(width_values):
                background.paste(triangles, (index * tile_size_width, height_row * tile_size_height))


        off_set_text = [(0, -40), (40, 0), (0, 40), (-40, 0)]

        for height_row, width_values in enumerate(self.q_table):
            for index, q_values in enumerate(width_values):
                max_value = max(q_values)
                for index_v, value in enumerate(q_values):
                    ImageDraw.Draw(background).text(
                        (index * tile_size_width + tile_size_width / 2.5 + off_set_text[index_v][0],
                         height_row * tile_size_height + tile_size_height / 2.5 + off_set_text[index_v][1]),
                        f"{round(value, 2)}", fill=(255, 255, 255))
                    if value == max_value:
                        if max_value != 0:
                            rect_color = (int(255 * value / max_value), 0, 0)
                        else:
                            rect_color = (0, 0, 0)

                        ImageDraw.Draw(background).rectangle(
                            [
                                (index * tile_size_width + tile_size_width / 2.5 + off_set_text[index_v][0] - 10,
                                 height_row * tile_size_height + tile_size_height / 2.5 + off_set_text[index_v][
                                     1] - 10),
                                (index * tile_size_width + tile_size_width / 2.5 + off_set_text[index_v][0] + 10,
                                 height_row * tile_size_height + tile_size_height / 2.5 + off_set_text[index_v][1] + 10)
                            ],
                            outline=rect_color
                        )



        cv2.imshow('Q-table', cv2.cvtColor(np.array(background), cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return self.q_table




class EpsilonSoftGreedyDoubleQPolicy(Policy):
    """Epsilon Soft greedy double Q policy."""

    def __init__(self, epsilon=0.9):
        """
        Create Epsilon Soft greedy double Q policy with parameters.

        :param epsilon: epsilon used in algorithm.
        """
        self.value_matrix = None
        self.epsilon = epsilon
        self.q_table_1 = None
        self.q_table_2 = None

    def decide_action(self, observation):
        """
        Decide action with highest value in q-table with a Epsilon change to take a random action.

        :param observation: observation is a dict containing information about the environment
        :return: Action chosen based on the observation
        """
        all_actions = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]

        if np.random.rand(1)[0] < self.epsilon:
            agent_pos = observation["agent_location"]
            chosen_action = np.argmax([x[0] + x[1] for x in zip(self.q_table_1[agent_pos[0]][agent_pos[1]], self.q_table_2[agent_pos[0]][agent_pos[1]])])
            return chosen_action
        else:
            return np.random.choice(all_actions)

    def visualise_q_table(self):
        """Visualise q table to img."""
        triangles = Image.open(textures_path / "baground_tiles_for_q.png")

        tile_size_width, tile_size_height = triangles.size

        maze = np.ndarray((4, 4))

        width = maze.shape[0] * tile_size_width
        height = maze.shape[1] * tile_size_height
        background = Image.new(mode="RGB", size=(width, height),color=(70, 0, 255))

        for height_row, width_values in enumerate(maze):
            for index, value in enumerate(width_values):
                background.paste(triangles, (index * tile_size_width, height_row * tile_size_height))

        off_set_text = [(0, -40), (40, 0), (0, 40), (-40, 0)]

        for height_row, width_values in enumerate(self.q_table_1):
            for index, q_values in enumerate(width_values):
                max_value = max([x[0] + x[1] for x in zip(self.q_table_2[height_row][index], q_values)])
                for index_v, value in enumerate(q_values):
                    value += self.q_table_2[height_row][index][index_v]
                    ImageDraw.Draw(background).text(
                        (index * tile_size_width + tile_size_width / 2.5 + off_set_text[index_v][0],
                         height_row * tile_size_height + tile_size_height / 2.5 + off_set_text[index_v][1]),
                        f"{round(value, 2)}", fill=(50, 0, 0, 50))
                    if value == max_value:
                        if max_value != 0:
                            rect_color = (int(255 * value / max_value), 0, 0)
                        else:
                            rect_color = (0, 0, 0)
                        ImageDraw.Draw(background).rectangle(
                            [
                                (index * tile_size_width + tile_size_width / 2.5 + off_set_text[index_v][0] - 10,
                                 height_row * tile_size_height + tile_size_height / 2.5 + off_set_text[index_v][
                                     1] - 10),
                                (index * tile_size_width + tile_size_width / 2.5 + off_set_text[index_v][0] + 10,
                                 height_row * tile_size_height + tile_size_height / 2.5 + off_set_text[index_v][1] + 10)
                            ],
                            outline=rect_color
                        )


        cv2.imshow('Q-table', cv2.cvtColor(np.array(background), cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return self.q_table_1, self.q_table_2

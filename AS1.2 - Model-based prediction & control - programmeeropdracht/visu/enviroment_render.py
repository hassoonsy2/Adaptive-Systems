from pathlib import Path
from PIL import Image, ImageDraw

textures_path = Path(__file__).parent / 'textures'


def render_background(environment):
    path = Image.open(textures_path / "grass.png")
    space = Image.open(textures_path / "space.png")
    highlight = Image.open(textures_path / "highlighter.png")
    exit_s = Image.open(textures_path / "objective_marker.png")

    tile_size_width, tile_size_height = path.size
    maze = environment.reward_map
    width, height = maze.shape[0] * tile_size_width, maze.shape[1] * tile_size_height
    background = Image.new(mode="RGB", size=(width, height))

    for y, row in enumerate(maze):
        for x, reward_value in enumerate(row):
            img = space if reward_value == -10 else path
            background.paste(img, (x * tile_size_width, y * tile_size_height), img)
            background.paste(highlight, (x * tile_size_width, y * tile_size_height), highlight)

    for y, row in enumerate(environment.reward_map):
        for x, rewards in enumerate(row):
            ImageDraw.Draw(background).text((x * tile_size_width + tile_size_width / 4,
                                             y * tile_size_height + tile_size_height / 2),
                                            f"R ={rewards}")

    for exit_maze in environment.end_coord:
        background.paste(exit_s,
                         (exit_maze[0] * tile_size_width + tile_size_width // 8,
                          exit_maze[1] * tile_size_height + tile_size_height // 8),
                         exit_s)

    return background


def render_in_step(environment):
    copy_background = environment.rendered_background.copy()
    agent_icon = Image.open(textures_path / "agent.png")

    tile_width = environment.rendered_background.width // environment.maze.shape[0]
    tile_size_width, tile_size_height = agent_icon.size

    ImageDraw.Draw(copy_background).text((5, 0), f"Time: {environment.sim_step}\nReward: {environment.total_reward}")

    action_to_string_dict = {0: 'up',
                             1: 'right',
                             2: 'down',
                             3: 'left',
                             4: 'stay',
                             None: 'None'}

    ImageDraw.Draw(copy_background).text((environment.rendered_background.width - 2 * tile_width, 0),
                                         f"Last action: {action_to_string_dict[environment.last_action_agent]}")

    if environment.agent.policy.visual_matrix is not None:
        for y, row in enumerate(environment.agent.policy.visual_matrix):
            for x, action in enumerate(row):
                ImageDraw.Draw(copy_background).text((x * tile_size_width + tile_size_width / 4,
                                                      y * tile_size_height + tile_size_height / 3),
                                                     f"A ={action}")

    copy_background.paste(agent_icon,
                          (environment.agent_location[1] * tile_size_width,
                           environment.agent_location[0] * tile_size_height),
                          agent_icon)

    return copy_background

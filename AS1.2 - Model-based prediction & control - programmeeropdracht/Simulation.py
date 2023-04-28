import cv2
import numpy as np
import sys
from maze import Maze
from policy import PureRandomPolicy ,ValueBasedPolicy
from agent import Agent


def run_simulation(environment, wait_key, window_name):
    done = False
    total_reward = 0
    observation = environment.get_state()

    while not done:
        if environment.visualize:
            render = environment.render()
            cv2.imshow(window_name, cv2.cvtColor(np.array(render), cv2.COLOR_BGR2RGB))
            cv2.waitKey(wait_key)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                environment.visualize = False
                break

        action = a1.get_action_from_policy(observation)
        observation, reward, done, info = environment.step(action)
        total_reward += reward

    return total_reward
def show_policy_selection_screen(window_name):
    # Define the screen size
    width, height = 800, 800

    # Create a black image
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Set the font and size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5

    # Define the text and positions
    text1 = 'Press 0 for Pure Random Policy'
    text2 = 'Press 1 for Value Based Policy'
    size1 = cv2.getTextSize(text1, font, font_scale, 2)[0]
    size2 = cv2.getTextSize(text2, font, font_scale, 2)[0]

    # Put the text on the image
    cv2.putText(img, text1, (width//2 - size1[0]//2, height//2 - size1[1]), font, font_scale, (255, 255, 255), 2)
    cv2.putText(img, text2, (width//2 - size2[0]//2, height//2 + size2[1]*2), font, font_scale, (255, 255, 255), 2)

    # Show the policy selection screen
    cv2.imshow(window_name, img)
    cv2.waitKey(1)

    return img


if __name__ == "__main__":
    window_name = 'sim'
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(window_name, 800, 800)

    # Show the policy selection screen and wait for user input
    show_policy_selection_screen(window_name)
    key = cv2.waitKey(0)

    # Check the user input and create the corresponding policy
    if key == ord('0'):
        policy = PureRandomPolicy()
    elif key == ord('1'):
        policy = ValueBasedPolicy()
    else:
        print("Invalid input. Exiting.")
        cv2.destroyAllWindows()
        sys.exit()

    # Close the policy selection screen
    cv2.destroyWindow(window_name)

    # Initialize the agent and environment
    a1 = Agent(policy)
    environment = Maze(a1, visualize=True)
    a1.value_iteration()

    if environment.visualize:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(window_name, 800, 800)

    # Run the simulation
    total_reward = run_simulation(environment, 0, window_name)

    if environment.visualize:
        render = environment.render()
        cv2.imshow(window_name, cv2.cvtColor(np.array(render), cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Print the total reward and simulation time
    print(f"{total_reward=}\ntime={environment.sim_step}")


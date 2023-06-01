import sys
import cv2
import numpy as np
from maze import Maze
from policy import PureRandomPolicy, ValueBasedPolicy, EpsilonSoftGreedyPolicy, EpsilonSoftGreedyDoubleQPolicy
from agent import Agent
from first_visit_mc import first_visit_mc
from temoral_difference_learning import tem_dif_ler
from first_visit_mc import on_policy_first_visit_mc_control
from sarsa import sarsa_tem_dif_ler
from q_learning import q_learning, double_q_learning
import os
# Define button colors
black = (0,0,0)
white = (255,255,255)

# Define dimensions
button_w = 630
button_h = 50

def draw_button(image, text, top_left, w, h, color):
    # Draw the button
    cv2.rectangle(image, top_left, (top_left[0]+w, top_left[1]+h), color, thickness=cv2.FILLED)
    # Add the text
    font_scale = 1
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_top_left = (top_left[0]+w//2-size[0]//2, top_left[1]+h//2+size[1]//2)
    cv2.putText(image, text, text_top_left, font, font_scale, black, thickness)

# Initialize a black image
img = np.zeros((850,850,3), np.uint8)

# Draw buttons for each policy
policies = [" 1 Monte-Carlo ", " 2 Temporal Difference Learning ", " 3 On policy first visit Monte-carlo control", " 4 On policy SARSA TD ", " 5 Q leanring", " 6 Double Q learning"]
for i, policy in enumerate(policies):
    draw_button(img, policy, (200,50+100*i), button_w, button_h, white)

# Display the image in a window
cv2.imshow('Policies', img)

# Wait for the user to press a key
key = cv2.waitKey(0)

# Based on the key pressed, select the policy
if key == ord('1'):
    print("Policy 1 selected.")
    sys.argv.append('0')
elif key == ord('2'):
    print("Policy 2 selected.")
    sys.argv.append('1')
elif key == ord('3'):
    print("Policy 3 selected.")
    sys.argv.append('2')
elif key == ord('4'):
    print("Policy 4 selected.")
    sys.argv.append('3')
elif key == ord('5'):
    print("Policy 5 selected.")
    sys.argv.append('4')
elif key == ord('6'):
    print("Policy 6 selected.")
    sys.argv.append('5')

cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        path = os.getcwd()
        # print(path)
        if 'AS2.2 - Model-free prediction & control - programmeeropdracht' not in path:
            os.chdir('AS2.2 - Model-free prediction & control - programmeeropdracht')

    # Creating environment
    policy_pr = PureRandomPolicy()
    a1 = Agent(policy_pr)

    policy_vb = ValueBasedPolicy()
    a2 = Agent(policy_vb)

    # Creating epsilon soft greedy policy
    policy_epsilon_greedy = EpsilonSoftGreedyPolicy()
    a3 = Agent(policy_epsilon_greedy)

    policy_epsilon_greedy_double_q = EpsilonSoftGreedyDoubleQPolicy()
    a4 = Agent(policy_epsilon_greedy_double_q)

    environment_pr = Maze(a1, visualize=False)
    environment_vb = Maze(a2, visualize=False)
    environment_eg = Maze(a3, visualize=False)
    environment_egdq = Maze(a4, visualize=False)

    # Creating variables that keep track of the simulation
    done = False
    total_reward = 0
    try:
        if int(sys.argv[1]) == 0:
            # Value function for value based policy
            a2.value_iteration()
            #a2.save_value_matrix('value_iteration_matrix.csv')

            # Load optimal value matrix
            #a2.load_value_matrix('value_iteration_matrix.csv')
            print(a2.policy.value_matrix)

            iterations = 10000
            discount_rate = 1
            exploring_starts = True
            print(f"Value based poly ;{iterations=}\t{discount_rate=}\t{exploring_starts=}\nOutcome\n")

            print(first_visit_mc(environment_vb,
                                 iterations=iterations,
                                 discount_rate=discount_rate,
                                 exploring_starts=exploring_starts), "\n")

            discount_rate = 0.9

            print(f"Value based poly ;{iterations=}\t{discount_rate=}\t{exploring_starts=}\nOutcome\n")

            print(first_visit_mc(environment_vb,
                                 iterations=iterations,
                                 discount_rate=discount_rate,
                                 exploring_starts=exploring_starts), "\n")

            discount_rate = 1

            print(f"Random based poly ;{iterations=}\t{discount_rate=}\t{exploring_starts=}\nOutcome\n")

            print(first_visit_mc(environment_pr,
                                 iterations=iterations,
                                 discount_rate=discount_rate,
                                 exploring_starts=exploring_starts), "\n")

            discount_rate = 0.9

            print(f"Random based poly ;{iterations=}\t{discount_rate=}\t{exploring_starts=}\nOutcome\n")

            print(first_visit_mc(environment_pr,
                                 iterations=iterations,
                                 discount_rate=discount_rate,
                                 exploring_starts=exploring_starts), "\n")

        if int(sys.argv[1]) == 1:
            # Value function for value based policy
            a2.value_iteration()
            #a2.save_value_matrix('value_iteration_matrix_value_based.csv')

            # Load optimal value matrix
            #a2.load_value_matrix('policy_saves/value_iteration_matrix.csv')
            # print(a2.policy.value_matrix)

            iterations = 10000
            discount_rate = 1
            alpha = 0.1
            exploring_starts = True
            print(
                f"Value based poly Temporal Difference Learning\n{iterations=}\t{discount_rate=}\t{alpha=}\t{exploring_starts=}\nOutcome\n")
            print(tem_dif_ler(environment_vb,
                              iterations=iterations,
                              discount_rate=discount_rate,
                              alpha=alpha,
                              exploring_starts=exploring_starts))
            discount_rate = 0.9
            print(
                f"Value based poly Temporal Difference Learning\n{iterations=}\t{discount_rate=}\t{alpha=}\t{exploring_starts=}\nOutcome\n")
            print(tem_dif_ler(environment_vb,
                              iterations=iterations,
                              discount_rate=discount_rate,
                              alpha=alpha,
                              exploring_starts=exploring_starts))

            discount_rate = 1
            alpha = 0.1
            exploring_starts = True
            print(
                f"Random based poly Temporal Difference Learning\n{iterations=}\t{discount_rate=}\t{alpha=}\t{exploring_starts=}\nOutcome\n")
            print(tem_dif_ler(environment_pr,
                              iterations=iterations,
                              discount_rate=discount_rate,
                              alpha=alpha,
                              exploring_starts=exploring_starts))

            discount_rate = 0.9
            print(
                f"Random based poly Temporal Difference Learning\n{iterations=}\t{discount_rate=}\t{alpha=}\t{exploring_starts=}\nOutcome\n")
            print(tem_dif_ler(environment_pr,
                              iterations=iterations,
                              discount_rate=discount_rate,
                              alpha=alpha,
                              exploring_starts=exploring_starts))

        if int(sys.argv[1]) == 2:
            iterations = 10000
            discount_rate = 1
            # discount_rate = 0.9
            exploring_starts = True
            epsilon = 0.9
            print(
                f"on policy control e soft greedy policy\n{iterations=}\t{discount_rate=}\t\t{exploring_starts=}\t{epsilon=}\nOutcome\n")
            print(on_policy_first_visit_mc_control(environment_eg,
                                                   iterations=iterations,
                                                   discount_rate=discount_rate,
                                                   exploring_starts=exploring_starts,
                                                   epsilon=epsilon))

        if int(sys.argv[1]) == 3:
            iterations = 10000
            discount_rate = 1
            # discount_rate = 0.9
            alpha = 0.1
            epsilon = 0.9
            exploring_starts = True
            print(
                f"Sarsa control Temporal Difference Learning\n{iterations=}\t{discount_rate=}\t{alpha=}\t{epsilon=}\t{exploring_starts=}\nOutcome\n")
            print(sarsa_tem_dif_ler(environment_eg,
                                    iterations=iterations,
                                    discount_rate=discount_rate,
                                    alpha=alpha,
                                    epsilon=epsilon,
                                    exploring_starts=exploring_starts))

        if int(sys.argv[1]) == 4:
            iterations = 50000
            # discount_rate = 1
            discount_rate = 0.9
            alpha = 0.1
            epsilon = 0.9
            exploring_starts = True
            print(
                f"Q-Learning\n{iterations=}\t{discount_rate=}\t{alpha=}\t{epsilon=}\t{exploring_starts=}\nOutcome\n")
            print(q_learning(environment_eg,
                             iterations=iterations,
                             discount_rate=discount_rate,
                             alpha=alpha,
                             epsilon=epsilon,
                             exploring_starts=exploring_starts))

        if int(sys.argv[1]) == 5:
            iterations = 50000
            discount_rate = 1
            # discount_rate = 0.9
            alpha = 0.1
            epsilon = 0.9
            exploring_starts = True
            print(
                f"DoubleQ-Learning\n{iterations=}\t{discount_rate=}\t{alpha=}\t{epsilon=}\t{exploring_starts=}\nOutcome\n")
            print(double_q_learning(environment_egdq,
                                    iterations=iterations,
                                    discount_rate=discount_rate,
                                    alpha=alpha,
                                    epsilon=epsilon,
                                    exploring_starts=exploring_starts))

    except IndexError:
        print("IndexError: Please enter a valid argument.")
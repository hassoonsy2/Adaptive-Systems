"""On policy first visit Monte-carlo control."""
import copy
import numpy as np


def on_policy_first_visit_mc_control(environment,
                                     iterations=10000,
                                     discount_rate=0.9,
                                     exploring_starts=False,
                                     epsilon=0.7):
    """
    On policy Monte Carlo control methods for updating given policy.

    Pseudo Code
    Input: a policy π to be evaluated
    Initialize:
        π ← an arbitrary ε-soft policy
        Q(s,a) ∈ R, arbitrarily, for all s ∈ S ∈ A(s)
        Returns(s, a) ← an empty list, for all s ∈ S ∈ A(s)

    Loop forever (for each episode):
        Generate an episode following π: S0,A0,R1, S1,A1,R2, . . . , ST−1,AT−1,RT
        G ← 0
        Loop for each step of episode, t = T −1, T −2, . . . , 0:
            G ← γG + Rt+1
            Unless St appears in S0, S1, . . . , St−1:
                Append G to Returns(St, At)
                Q(St, At) ← average(Returns(St, At))
                A* ← argmax a Q(St,a)                   (with ties broken arbitrarily)
                for all a ∈ A(St):
                                1 - ε + ε/|A(St)|   if a = A*
                    π(a|St) ←
                                ε/|A(St)|           if a ≠ A*

    :param environment: Environment of the simulation contains the agent with policy
    :param iterations: Loop amount for creating episodes
    :param discount_rate: Discount value used in algorithm
    :param exploring_starts: Enable or disable exploring starts
    :param epsilon: Parameter for E-soft policy
    :return:
    """
    #    Initialize:
    #    π ← an arbitrary ε-soft policy
    #    Q(s,a) ∈ R, arbitrarily, for all s ∈ S ∈ A(s)
    #    Returns(s, a) ← an empty list, for all s ∈ S ∈ A(s)

    environment.agent.policy.epsilon = epsilon

    dict_of_states = {}
    # Q table to policy
    q_table = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] for x in np.zeros_like(environment.maze)]
    environment.agent.policy.q_table = q_table

    array_estimates_policy = copy.copy(environment.maze)
    # Iterate over y and x
    for index_y, x in enumerate(array_estimates_policy):
        for index_x, _ in enumerate(x):
            # all_action = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
            state = (index_y, index_x)
            dict_of_states[state] = {"action_value": [0, 0, 0, 0],
                                     "rewards": [[], [], [], []],
                                     "average": [0, 0, 0, 0]}

    # Loop forever (for each episode):
    #     Generate an episode following π: S0,A0,R1, S1,A1,R2, . . . , ST−1,AT−1,RT
    counter = 0
    episodes_dict = {}

    for i in range(iterations):
        episode_log = []
        environment.reset(random_start=exploring_starts)

        total_reward = 0
        # Get first observation for loop
        first_observation = environment.get_state()
        action = environment.agent.get_action_from_policy(first_observation)
        observation, reward, _, info = environment.step(action)
        total_reward += reward

        episode_log.append([first_observation['agent_location'], action, reward])

        while not environment.done:
            counter += 1
            # Decide an action according to the observation
            action = environment.agent.get_action_from_policy(observation)
            last_observation = observation
            # Take action in the world
            observation, reward, _, info = environment.step(action)
            episode_log.append([last_observation['agent_location'], action, reward])

            # Counting reward
            total_reward += reward
        episodes_dict[i] = episode_log
        episodes_dict[i].append([total_reward, environment.sim_step])

        # G ← 0
        big_g = 0

        # Loop for each step of episode, t = T −1, T −2, . . . , 0:
        inverted_episode_log = episode_log[::-1][1:]
        for index, step_info in enumerate(inverted_episode_log):
            # Step info[0] = State
            # Step info[1] = Action
            # Step info[2] = Reward
            state_info = step_info[0]
            action_info = step_info[1]
            reward_info = step_info[2]

            # G ← γG + Rt+1
            big_g = discount_rate * big_g + reward_info

            # Unless St appears in S0, S1, . . . , St−1:
            if not state_info in [x[0] for x in inverted_episode_log[index + 1:]]:
                # Append G to Returns(St, At)
                dict_of_states[step_info[0]]['rewards'][action_info].append(big_g)

                # Q(St, At) ← average(Returns(St, At))
                dict_of_states[step_info[0]]['average'][action_info] = np.average(
                    dict_of_states[state_info]['rewards'][action_info])

                # A* ← argmax a Q(St,a)                   (with ties broken arbitrarily)
                A_star = (step_info[0], np.argmax(dict_of_states[step_info[0]]['average']))

                #         for all a ∈ A(St):
                #                         1 - ε + ε/|A(St)|   if a = A*
                #             π(a|St) ←
                #                         ε/|A(St)|           if a ≠ A*
                for action_index, every_a in enumerate(dict_of_states[step_info[0]]['average']):
                    if index == A_star[1]:
                        dict_of_states[step_info[0]]['average'][action_index] = 1 - epsilon + epsilon * dict_of_states[step_info[0]]['average'][action_index]
                    else:
                        dict_of_states[step_info[0]]['average'][action_index] = epsilon * dict_of_states[step_info[0]]['average'][action_index]

                environment.agent.policy.q_table[state_info[0]][state_info[1]][action_info] = \
                    dict_of_states[step_info[0]]['average'][action_info]

    return environment.agent.policy.visualise_q_table()



"""First visit Monte-carlo evaluation."""



def first_visit_mc(environment, iterations=10000, discount_rate=0.9, exploring_starts=False):
    """
    First Monte Carlo methods for learning the state-value function for a given policy.

    Pseudo Code
    Input: a policy π to be evaluated
    Initialize:
        V (s) ∈ R, arbitrarily, for all s ∈ S
        Returns(s) ← an empty list, for all s ∈ S

    Loop forever (for each episode):
        Generate an episode following π: S0,A0,R1, S1,A1,R2, . . . , ST−1,AT−1,RT
        G ← 0
        Loop for each step of episode, t = T −1, T −2, . . . , 0:
            G ← γG + Rt+1
            Unless St appears in S0, S1, . . . , St−1:
                Append G to Returns(St)
                V (St) ← average(Returns(St))

    :param environment: Environment of the simulation contains the agent with policy
    :param iterations: Loop amount for creating episodes
    :param discount_rate: Discount value used in algorithm
    :param exploring_starts: Enable or disable exploring starts
    :return: Value matrix of given policy in environment given
    """
    #    Initialize:
    #    V (s) ∈ R, arbitrarily, for all s ∈ S
    #    Returns(s) ← an empty list, for all s ∈ S

    dict_of_states = {}
    array_estimates_policy = copy.copy(environment.maze)
    # Iterate over y and x
    for index_y, x in enumerate(array_estimates_policy):
        for index_x, _ in enumerate(x):
            # all_action = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
            state = (index_y, index_x)
            dict_of_states[state] = {"action_value": [0, 0, 0, 0], "rewards": [], "average": 0}

    # Loop forever (for each episode):
    #     Generate an episode following π: S0,A0,R1, S1,A1,R2, . . . , ST−1,AT−1,RT
    counter = 0
    episodes_dict = {}

    for i in range(iterations):
        episode_log = []
        environment.reset(random_start=exploring_starts)

        total_reward = 0
        # Get first observation for loop
        first_observation = environment.get_state()
        action = environment.agent.get_action_from_policy(first_observation)
        observation, reward, _, info = environment.step(action)
        total_reward += reward

        episode_log.append([first_observation['agent_location'], action, reward])

        while not environment.done:
            counter += 1
            # Decide an action according to the observation
            action = environment.agent.get_action_from_policy(observation)
            last_observation = observation
            # Take action in the world
            observation, reward, _, info = environment.step(action)
            episode_log.append([last_observation['agent_location'], action, reward])

            # Counting reward
            total_reward += reward
        episodes_dict[i] = episode_log
        episodes_dict[i].append([total_reward, environment.sim_step])

        # G ← 0
        big_g = 0

        # Loop for each step of episode, t = T −1, T −2, . . . , 0:
        inverted_episode_log = episode_log[::-1][1:]
        for index, step_info in enumerate(inverted_episode_log):
            # Step info[0] = State
            # Step info[1] = Action
            # Step info[2] = Reward
            # G ← γG + Rt+1
            big_g = discount_rate * big_g + step_info[2]
            # Unless St appears in S0, S1, . . . , St−1:
            if not step_info[0] in [x[0] for x in inverted_episode_log[index + 1:]]:
                # Append G to Returns(St)
                dict_of_states[step_info[0]]['rewards'].append(big_g)
                # V (St) ← average(Returns(St))
                dict_of_states[step_info[0]]['average'] = np.average(dict_of_states[step_info[0]]['rewards'])

                # Qn+1  = NewEstimate OldEstimate + StepSize (Target - oldestimate)
                """
                old_estimate = dict_of_states[step_info[0]]['average']
                step_size = 1 / (index + 1)
                target = step_info[2]

                dict_of_states[step_info[0]]['average'] = old_estimate + (step_size * (target - old_estimate))
                """

        value_matrix = copy.copy(environment.maze)
        for key in dict_of_states.keys():
            value_matrix[key] = round(dict_of_states[key]['average'], 2)
    return value_matrix
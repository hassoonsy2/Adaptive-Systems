"""Sarsa on policy Temporal Difference control."""
import numpy as np


def sarsa_tem_dif_ler(environment, iterations=1000, discount_rate=0.9, alpha=0.1, exploring_starts=False, epsilon=0.9):
    """
    Policy control using SARSA temporal difference.

    Pseudo Code
    Algorithm parameter: step size α ∈ (0,1], ε > 0
    Initialize Q(s,a), for all s ∈ S+,a ∈ A(s), arbitrarily except that V (terminal, *) = 0

    Loop for each step of episode
        Initialize S
        Choose A from S using policy derived from Q (e.g., ε-greedy)
            Loop for each step of episode:
            Take action A, observe R, S'
            Choose A' from S' using policy derived from Q (e.g., ε-greedy)
            Q(S,A) ← Q(S,A) + α (R + γQ(S',A') - Q(S,A))
            S ← S'; A ← A'
        until s is terminal

    :param environment: Environment of the simulation contains the agent with policy
    :param iterations: Loop amount for creating episodes
    :param discount_rate: Discount value used in algorithm
    :param alpha: alpha used in algorithm
    :param exploring_starts: Enable or disable exploring starts
    :param epsilon: Parameter for E-soft policy
    :return: Value matrix of given policy in environment given
    """
    q_table = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] for _ in np.zeros_like(environment.maze)]
    environment.agent.policy.q_table = q_table
    environment.agent.policy.epsilon = epsilon

    for i in range(iterations):
        environment.reset(random_start=exploring_starts)
        total_reward = 0

        # Initialize S
        state = environment.get_state()
        # Choose A from S using policy derived from Q (e.g., ε-greedy)
        action = environment.agent.get_action_from_policy(state)

        while not environment.done:
            # Take action A, observe R, S'
            state_prime, reward, _, _ = environment.step(action)

            # Choose A' from S' using policy derived from Q (e.g., ε-greedy)
            action_prime = environment.agent.get_action_from_policy(state_prime)

            # Q(S,A) ← Q(S,A) + α (R + γQ(S',A') - Q(S,A))
            # Q(S,A)        ←  Q             (                 S                                    ,A)       +   α     (    R  + γ                                        Q       (                            S'                                    ,A'          ) -                           Q      (                            S                          ,A))
            environment.agent.policy.q_table[state['agent_location'][0]][state['agent_location'][1]][action] += alpha * (reward + discount_rate * environment.agent.policy.q_table[state_prime['agent_location'][0]][state_prime['agent_location'][1]][action_prime] - environment.agent.policy.q_table[state['agent_location'][0]][state['agent_location'][1]][action])

            total_reward += reward

            # S ← S'; A ← A'
            state = state_prime
            action = action_prime

    return environment.agent.policy.visualise_q_table()
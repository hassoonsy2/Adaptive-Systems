import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from Agent2 import Agent
from EpsilonGreedyPolicy import EpsilonGreedyPolicy
from Memory import Memory
from Transition import Transition

def train(episodes: int,
          batch_size: int,
          update_network_N: int = 10,
          tau: float = 0.01,
          gamma: float = 0.99,
          learning_rate: float = 0.0001,
          model_middle_layer_size: int = 256,
          steps_per_episode: int = 2000,
          memory_size: int = 10000,
          # new_network: bool = False,
          base_epsilon: float = 0.9,
          decay_factor: float = 0.999,
          minimal_epsilon: float = 0.001
          ):
    env = gym.make('LunarLander-v2',render_mode="human")

    policy = EpsilonGreedyPolicy(env, base_epsilon, decay_factor, minimal_epsilon)
    total_actions = env.action_space.n

    state,_ = env.reset()
    state = np.array(state)



    observation_length = len(state)


    agent = Agent(policy, tau=tau, batch_size=batch_size, gamma=gamma, learning_rate=learning_rate,
                  model_input_size=observation_length, model_output_size=total_actions, model_middle_layer_size=model_middle_layer_size)

    # if not new_network:
    #     agent.load_model('primary_name', 'target_name')

    memory = Memory(size=memory_size)

    update_network_counter = 1

    for episode in range(episodes):

        done = False
        iteration = 0
        total_reward = 0

        state,_ = env.reset()
        state = np.array(state)

        for step in range(steps_per_episode):
            update_network_counter += 1
            iteration += 1

            action = agent.get_action(state)

            next_state, reward, done, info, _ = env.step(action)
            total_reward += reward

            memory.append_record(Transition(state, action, reward, done, next_state))

            state = next_state

            if memory.get_deque_len() >= batch_size:
                if update_network_counter % update_network_N == 0:
                    batch = memory.sample(batch_size)

                    agent.train(batch)
            if done:
                break

        policy.epsilon_decay()

        if episode % 10 == 0:
            print(f'Episode: {episode} - Total Reward: {total_reward} - Average Reward: {total_reward/iteration} - Epsilon: {policy.epsilon}')
            print("saving")
            agent.approximator.save_network(agent.primary_network, agent.target_network)

if __name__ == "__main__":
    train(episodes=5000,
          batch_size=64,
          update_network_N=4,
          tau=0.001,
          gamma=0.99,
          learning_rate=0.1,
          model_middle_layer_size=256,
          memory_size=50000)

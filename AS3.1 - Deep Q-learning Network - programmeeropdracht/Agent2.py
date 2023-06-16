import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from SaveLoadNetwork import SaveLoadNetwork


class Agent:
    """Agent class used for Double Deep Q-Learning."""

    def __init__(self, policy,
                 tau=0.001,
                 batch_size=64,
                 gamma=0.99,
                 learning_rate=0.0001,
                 model_input_size=8,
                 model_output_size=4,
                 model_middle_layer_size=256):
        """Initializes the agent class

        :param policy: The agent's policy
        :param tau: Variable for algorithm
        :param batch_size: Memory batch size for algorithm
        :param gamma: Discount value for algorithm
        :param learning_rate: learning rate factor for optimizer
        :param model_input_size: Number of input nodes
        :param model_output_size: Number of output nodes
        :param model_middle_layer_size: Number of hidden layer nodes
        """

        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.policy = policy
        self.approximator = SaveLoadNetwork()
        self.primary_network, self.target_network = self.approximator.create_networks(input_size=model_input_size,
                                                                                      output_size=model_output_size,
                                                                                      middle_layer_size=model_middle_layer_size)


        self.primary_network.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        self.target_network.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

    def load_model(self, primary_nn_name='default_primary_name', target_nn_name='default_target_name'):
        """Load TensorFlow model from models folder.

        :param primary_nn_name: Primary network file name
        :param target_nn_name: Target network file name
        """
        self.primary_network = tf.keras.models.load_model(primary_nn_name)
        self.target_network = tf.keras.models.load_model(target_nn_name)

    def get_action(self, state):
        """Get action from policy.

        :param state: Environment state
        :return: Chosen action
        """
        return self.policy.select_action(state, self.primary_network)

    def train(self, train_batch):
        """Train a network with the Double Deep Q-Learning algorithm.

        :param train_batch: Set of transitions"""

        self.approximator.train_network(train_batch, self.primary_network, self.target_network, self.gamma)
        self.copy_model()

    def copy_model(self):
        """Copy primary model weight to target model with tau value to update just a small bit."""

        primary_weights = self.primary_network.get_weights()
        target_weights = self.target_network.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = self.tau * primary_weights[i] + (1 - self.tau) * target_weights[i]

        self.target_network.set_weights(target_weights)

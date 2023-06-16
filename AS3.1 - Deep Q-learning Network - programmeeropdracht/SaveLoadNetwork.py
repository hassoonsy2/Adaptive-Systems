import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses
import os


class SaveLoadNetwork:
    """Class for saving, loading, creating and training a Double Q Neural Network."""

    def __init__(self):
        """Create class variables for double deep learning."""
        self.q_network_1 = None
        self.q_network_2 = None
        self.model_path = os.path.dirname(os.getcwd()) + '/AS3.1 - Deep Q-learning Network - programmeeropdracht/'
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = losses.MeanSquaredError()

    def set_optimizer(self, learning_rate: float):
        """Set the optimizer for the model.

        :param learning_rate: Learning rate factor
        """
        self.optimizer = optimizers.Adam(learning_rate)

    def save_network(self,
                     primary_nn_name: str = 'default_primary_name',
                     target_nn_name: str = 'default_target_name'):
        """Save the networks used for double Q-learning.

        :param primary_nn_name: Primary network file name
        :param target_nn_name: Target network file name
        """
        model_path = self.model_path

        # Save primary network
        primary_nn_name = "default_primary_name"
        target_nn_name = "default_target_name"
        self.q_network_1.save(os.path.join(model_path, primary_nn_name + ".h5"))

        # Save target network
        self.q_network_2.save(os.path.join(model_path, target_nn_name + ".h5"))

    def load_network(self, primary_nn_name: str = 'default_primary_name',
                     target_nn_name: str = 'default_target_name'):
        """Load networks used for double Q-learning.

        :param primary_nn_name: Primary network file name
        :param target_nn_name: Target network file name
        """
        model_path = self.model_path

        # Load primary network
        self.q_network_1 = models.load_model(os.path.join(model_path, primary_nn_name + ".h5"))

        # Load target network
        self.q_network_2 = models.load_model(os.path.join(model_path, target_nn_name + ".h5"))

    def train_network(self, train_batch: object, primary_network, target_network, gamma: float):
        """Trains network.

        :param train_batch: Set of transitions
        :param primary_network: The primary Q network
        :param target_network: The target Q network
        :param gamma: Discount value for algorithm
        """

        state_batch = tf.stack([x[0] for x in train_batch])
        action_batch = tf.stack([x[1] for x in train_batch])
        reward_batch = tf.stack([x[2] for x in train_batch])
        done_batch = tf.stack([x[3] for x in train_batch])
        next_obs_batch = tf.stack([x[4] for x in train_batch])

        with tf.GradientTape() as tape:
            next_q_values = primary_network(next_obs_batch)
            next_q_values_target = target_network(next_obs_batch)

            q_star = tf.stack([tf.cast(reward, tf.double) if done else
                               tf.cast(reward, tf.double) + gamma * tf.cast(q_target[tf.argmax(next_q_pred).numpy()],
                                                                            tf.double)
                               for reward, next_q_pred, done, q_target in
                               zip(reward_batch, next_q_values, done_batch, next_q_values_target)
                               ])

            current_q_values = primary_network(state_batch)
            chosen_q = tf.stack([x[y] for x, y in zip(current_q_values, action_batch)])
            loss = self.loss_fn(q_star, chosen_q)

        grads = tape.gradient(loss, primary_network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, primary_network.trainable_weights))

    def create_network(self, input_size: int = 8, output_size: int = 4, middle_layer_size: int = 12):
        """Create neural network for the double Q learning.

        The network is build out of 4 layers in total from
        which 2 are hidden. This network is the primary network.

        :param input_size: Input size for the first layer
        :param output_size: Output size for the last layer
        :param middle_layer_size: Middle layer size of the first hidden layer
        """
        # Check if middle layer is smaller then input size and correct it
        if middle_layer_size < input_size:
            middle_layer_size = int(round(input_size * 1.5))
            print(
                f"Middle layer is smaller then input size and this is not ideal. check middle layer to size {middle_layer_size} ")

        # Calculate the output size of the 2e middle layer to be a bit smaller then the previous
        middle_layer_output = int(round(middle_layer_size // 1.25))

        model = models.Sequential([
            layers.Dense(middle_layer_size, activation='relu', input_shape=(input_size,)),
            layers.Dense(middle_layer_output, activation='relu'),
            layers.Dense(output_size)
        ])
        return model

    def create_networks(self, input_size: int = 8, output_size: int = 4, middle_layer_size: int = 12):
        self.q_network_1 = self.create_network(input_size, output_size, middle_layer_size)
        self.q_network_2 = self.create_network(input_size, output_size, middle_layer_size)
        return self.q_network_1, self.q_network_2

    def get_network_info(self):
        """Print information about the networks used for double deep Q- Learning."""
        print(f"Primary network =\n{self.q_network_1.summary()}\nTarget network =\n{self.q_network_2.summary()}")

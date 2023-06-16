import random
from collections import deque

class Memory:
    """Class file for memory used by double deep Q-Learning."""

    def __init__(self, size: int):
        """Initialize memory class.

        :param size: Maximum memory size
        """

        self.size = size
        self.transition_deque = deque(maxlen=size)

    def sample(self, batch_size: int = 10):
        """Return a random memory sample.

        :param batch_size: Batch size of random memory sample
        :return: A random memory sample
        """

        return random.sample(self.transition_deque, batch_size)

    def append_record(self, record: object):
        """Record a new memory record.

        :param record: A memory record
        """

        self.transition_deque.append(record)

    def get_deque_len(self):
        """Return length of the deque.

        :return: Deque length
        """

        return len(self.transition_deque)

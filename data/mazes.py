import numpy as np
import collections


class Mazes(object):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, data_root='../data/mazes/np_mazes_train.npy', seq_len=48, image_size=64):
        self.path = data_root
        self.seq_len = seq_len
        self.image_size = image_size
        self.seed_is_set = False  # multi threaded loading
        self.channels = 3
        self.sample_size = 900
        self.counter = 0
        self.dataset = None

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.sample_size

    def load_dataset(self):
        training_data = np.load(self.path) / 255.0
        # input shape [n, h, seq_len, w, channels]
        training_data = np.transpose(training_data, (0, 2, 1, 3, 4))
        training_data = np.transpose(training_data, (0, 1, 2, 4, 3))
        training_data = np.transpose(training_data, (0, 1, 3, 2, 4))
        training_data = np.transpose(training_data, (0, 2, 1, 3, 4))
        self.sample_size = training_data.shape[0]
        training_data = np.reshape(training_data, [self.sample_size, self.channels, self.seq_len,
                                                   self.image_size, self.image_size])
        self.dataset = training_data

    def __getitem__(self, index):
        self.set_seed(index)
        self.load_dataset()
        x = self.dataset[int(self.counter % self.sample_size)]
        self.counter += 1
        return x

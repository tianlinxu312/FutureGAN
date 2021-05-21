import numpy as np
#from torchvision import datasets, transforms


class MovingMNIST(object):
    
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root='../data/mmnist/', seq_len=20, num_digits=2,
                 image_size=64, deterministic=True):
        self.path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits  
        self.image_size = image_size 
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1
        self.sample_size = 8000
        self.counter = 0
        self.dataset = None

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.seq_len

    def load_dataset(self):
        training_data = np.load(self.path) / 255.0
        training_data = np.transpose(training_data, (1, 0, 2, 3))
        training_data = np.reshape(training_data, [self.sample_size, self.seq_len, self.image_size, self.image_size, 1])
        self.dataset = training_data

    def __getitem__(self, index):
        self.set_seed(index)
        self.load_dataset()
        x = self.dataset[int(self.counter % self.sample_size)]
        self.counter += 1
        return x

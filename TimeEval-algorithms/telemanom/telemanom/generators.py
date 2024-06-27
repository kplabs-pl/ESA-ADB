import random
import numpy as np
random.seed(1992)
np.random.seed(1992)
from tensorflow.keras.utils import Sequence


class TelemanomGenerator(Sequence):
    def __init__(self, data: np.array, train_means, train_stds, target_channel_index: int = 0, window_size: int = 250, prediction_window_size: int = 10, batch_size: int = 1, shuffle: bool = True):
        self.data = data
        self.train_means = train_means
        self.train_stds = train_stds
        self.target_channel_index = target_channel_index
        self.window_size = window_size
        self.prediction_window_size = prediction_window_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Samples are indexed by fragment number and index of the sample inside the fragment
        self.indices = []
        for i, arr in enumerate(self.data):
            self.nb_channels = arr.shape[-1]
            for j in range(len(arr) - self.window_size - self.prediction_window_size):
                self.indices.append((i, j))
        self.nb_samples = len(self.indices)

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.nb_samples / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, self.nb_samples)
        indices = self.indices[start:end]

        # If needed - fill last batch with random indices. This is crucial when using BatchNorm
        if self.shuffle and start + self.batch_size > self.nb_samples:
            nb_to_add = self.batch_size - len(indices)
            indices_to_add = np.array(self.indices)[np.random.choice(start, nb_to_add, replace=False)]
            indices = np.concatenate((indices, indices_to_add))

        input_data = []
        output_data = []
        for i, j in indices:
            window_end = j + self.window_size
            input_data.append((self.data[i][j:window_end] - self.train_means) / self.train_stds)
            output_data.append((self.data[i][window_end:window_end + self.prediction_window_size, self.target_channel_index] - self.train_means[self.target_channel_index]) / self.train_stds[self.target_channel_index])

        return np.array(input_data), np.array(output_data)

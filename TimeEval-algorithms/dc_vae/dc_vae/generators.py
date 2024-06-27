import random
import numpy as np
random.seed(1992)
np.random.seed(1992)
from tensorflow.keras.utils import Sequence


class DcVaeGenerator(Sequence):
    def __init__(self, data: np.array, train_means, train_stds, input_channel_indices: list, target_channels_indices: list, window_size: int = 250,
                 batch_size: int = 1, shuffle: bool = True,
                 prediction_mode: bool = False):
        self.data = data
        self.train_means = train_means
        self.train_stds = train_stds
        self.input_channel_indices = input_channel_indices
        self.target_channels_indices = target_channels_indices
        self.same_input_and_output = (input_channel_indices == target_channels_indices)
        self.window_size = window_size
        self.batch_size = batch_size
        self.prediction_mode = prediction_mode
        self.shuffle = shuffle

        # Sample indices are defined by fragment number and index of the sample inside the fragment
        self.indices = []
        for i, arr in enumerate(self.data):
            for j in range(len(arr) - self.window_size + 1):
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

        # If needed - fill last batch with random indices during training. This is important when using BatchNorm
        if self.shuffle and start + self.batch_size > self.nb_samples:
            nb_to_add = self.batch_size - len(indices)
            indices_to_add = np.array(self.indices)[np.random.choice(start, nb_to_add, replace=False)]
            indices = np.concatenate((indices, indices_to_add))

        input_data = []
        output_data = []
        for i, j in indices:
            window_end = j + self.window_size
            input_data.append(((self.data[i][j:window_end, self.input_channel_indices] - self.train_means[self.input_channel_indices]) / self.train_stds[self.input_channel_indices]).astype(np.float32))
            if not self.same_input_and_output:
                output_data.append(((self.data[i][j:window_end, self.target_channels_indices] - self.train_means[self.target_channels_indices]) / self.train_stds[self.target_channels_indices]).astype(np.float32))

        if self.same_input_and_output:
            output_data = input_data

        return [np.array(input_data), np.array(output_data)]

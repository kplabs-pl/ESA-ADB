import numpy as np
import os
import logging
from .generators import TelemanomGenerator
from numpy import random

logger = logging.getLogger('telemanom-esa')


class Channel:
    def __init__(self, config):
        """
        Load and reshape channel values (predicted and actual).

        Args:
            config (obj): Config object containing parameters for processing

        Attributes:
            id (str): channel id
            config (obj): see Args
            X_train (arr): training inputs with dimensions
                [timesteps, l_s, input dimensions)
            X_test (arr): test inputs with dimensions
                [timesteps, l_s, input dimensions)
            y_train (arr): actual channel training values with dimensions
                [timesteps, n_predictions, 1)
            y_test (arr): actual channel test values with dimensions
                [timesteps, n_predictions, 1)
            train (arr): train data loaded from .npy file
            test(arr): test data loaded from .npy file
        """

        self.config = config
        self.id = config.target_channels
        self.generator_train = None
        self.generator_val = None
        self.generator_test = None
        self.y_hat = None

    def shape_data(self, arr, binary_channels_mask, channels_minimums, channels_maximums, train=True):
        """Shape raw input streams for ingestion into LSTM. config.l_s specifies
        the sequence length of prior timesteps fed into the model at
        each timestep t.

        Args:
            arr (np array): array of input streams with
                dimensions [fragment number, timesteps, input dimensions]
            train (bool): If shaping training data, this indicates
                data can be shuffled
        """

        if train:
            if isinstance(arr, list):
                train_data = arr[0]
                val_data = arr[1]
            else:
                total_nb_samples = sum(len(a) for a in arr)
                nb_train_samples = round((1 - self.config.validation_split) * total_nb_samples)
                start_idx, end_idx = 0, 0
                train_data, val_data = [], []
                for a in arr:
                    end_idx += len(a)
                    if start_idx < nb_train_samples and end_idx < nb_train_samples:
                        train_data.append(a)
                    elif start_idx > nb_train_samples and end_idx > nb_train_samples:
                        val_data.append(a)
                    else:
                        train_data.append(a[:(nb_train_samples - start_idx)])
                        val_data.append(a[(nb_train_samples - start_idx):])
                    start_idx = end_idx
                train_data, val_data = np.array(train_data, dtype=object), np.array(val_data, dtype=object)

            train_means = np.mean(np.concatenate(train_data), axis=0)
            train_means = np.where(binary_channels_mask, channels_minimums, train_means)  # normalize binary channels
            np.savetxt(self.config.meansOutput, train_means)

            train_stds = np.std(np.concatenate(train_data).astype(float), axis=0)
            train_stds = np.where(binary_channels_mask, channels_maximums - channels_minimums, train_stds)  # normalize binary channels
            train_stds = np.where(train_stds == 0, 1, train_stds)  # do not divide constant signals by zero
            np.savetxt(self.config.stdsOutput, train_stds)

            self.generator_train = TelemanomGenerator(train_data, train_means, train_stds, self.config.input_channel_indices, self.config.target_channel_indices, self.config.window_size, self.config.prediction_window_size,
                                                      self.config.batch_size, shuffle=True)
            self.generator_val = TelemanomGenerator(val_data, train_means, train_stds, self.config.input_channel_indices, self.config.target_channel_indices, self.config.window_size, self.config.prediction_window_size,
                                                    self.config.batch_size, shuffle=False)
        else:
            train_means = np.atleast_1d(np.loadtxt(self.config.meansOutput))
            train_stds = np.atleast_1d(np.loadtxt(self.config.stdsOutput))

            self.generator_test = TelemanomGenerator(arr, train_means, train_stds, self.config.input_channel_indices, self.config.target_channel_indices, self.config.window_size, self.config.prediction_window_size,
                                                     self.config.batch_size, shuffle=False, prediction_mode=True)

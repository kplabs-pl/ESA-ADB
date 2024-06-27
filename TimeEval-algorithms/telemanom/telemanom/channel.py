import numpy as np
import os
import logging
from .generators import TelemanomGenerator

logger = logging.getLogger('telemanom')


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
        self.id = config.target_channel
        self.generator_train = None
        self.generator_val = None
        self.generator_test = None
        self.y_hat = None
        self.train = None
        self.test = None

    def shape_data(self, arr, train=True):
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
            train_data = arr
            if isinstance(train_data, list):
                train_data = arr[0]
                val_data = arr[1]
            else:
                total_nb_samples = sum(len(a) for a in train_data)
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
            train_stds = np.std(np.concatenate(train_data).astype(float), axis=0)
            np.savetxt(self.config.meansOutput, train_means)
            np.savetxt(self.config.stdsOutput, train_stds)

            self.generator_train = TelemanomGenerator(train_data, train_means, train_stds, self.config.target_channel, self.config.window_size, self.config.prediction_window_size,
                                                      self.config.batch_size, shuffle=True)
            self.generator_val = TelemanomGenerator(val_data, train_means, train_stds, self.config.target_channel, self.config.window_size, self.config.prediction_window_size,
                                                    self.config.batch_size, shuffle=False)
        else:
            train_means = np.atleast_1d(np.loadtxt(self.config.meansOutput))
            train_stds = np.atleast_1d(np.loadtxt(self.config.stdsOutput))

            self.generator_test = TelemanomGenerator(arr, train_means, train_stds, self.config.target_channel, self.config.window_size, self.config.prediction_window_size,
                                                     self.config.batch_size, shuffle=False)

    def load_data(self):
        """
        Load train and test data from local.
        """
        try:
            self.train = np.load(os.path.join("data", "train", "{}.npy".format(self.id)))
            self.test = np.load(os.path.join("data", "test", "{}.npy".format(self.id)))

        except FileNotFoundError as e:
            logger.critical(e)
            logger.critical("Source data not found, may need to add data to repo: <link>")

        self.shape_data(self.train)
        self.shape_data(self.test, train=False)

    def set_data(self, data: np.ndarray, train: bool):
        if train:
            self.train = data
        else:
            self.test = data

        self.shape_data(data, train=train)

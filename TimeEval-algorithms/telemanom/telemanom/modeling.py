from keras.models import Sequential, load_model
from keras.callbacks import History, EarlyStopping, Callback, ModelCheckpoint
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from typing import Optional
import numpy as np
import os
import logging

# suppress tensorflow CPU speedup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger('telemanom')


class Model:
    def __init__(self, config, run_id, channel, model_path: Optional[os.PathLike] = None):
        """
        Loads/trains RNN and predicts future telemetry values for a channel.

        Args:
            config (obj): Config object containing parameters for processing
                and model training
            run_id (str): Datetime referencing set of predictions in use
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Attributes:
            config (obj): see Args
            chan_id (str): channel id
            run_id (str): see Args
            y_hat (arr): predicted channel values
            model (obj): trained RNN model for predicting channel values
        """

        self.config = config
        self.chan_id = channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None
        self.model_path = model_path or os.path.join("data", self.config.use_id, "models", self.chan_id + '.h5')

        if not self.config.train:
            try:
                self.load()
            except FileNotFoundError:
                logger.warning('Training new model, couldn\'t find existing '
                               'model at {}'.format(self.model_path))
                self.train_new(channel)
                self.save()
        else:
            self.train_new(channel)
            self.save()

    def load(self):
        """
        Load model for channel.
        """

        logger.info('Loading pre-trained model')
        self.model = load_model(self.model_path)

    def train_new(self, channel):
        """
        Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        cbs = [History(),
               EarlyStopping(monitor='val_loss',
                             patience=self.config.patience,
                             min_delta=self.config.min_delta,
                             verbose=0),
               ModelCheckpoint(filepath=self.model_path,
                               verbose=1,
                               save_weights_only=False,
                               period=1)
               ]

        self.model = Sequential()

        self.model.add(LSTM(
            self.config.layers[0],
            input_shape=(None, channel.generator_train.nb_channels),
            return_sequences=True))
        self.model.add(Dropout(self.config.dropout))

        self.model.add(LSTM(
            self.config.layers[1],
            return_sequences=False))
        self.model.add(Dropout(self.config.dropout))

        self.model.add(Dense(
            self.config.prediction_window_size))
        self.model.add(Activation('linear'))

        self.model.compile(loss=self.config.loss_metric,
                           optimizer=self.config.optimizer)

        self.max_steps_per_epoch = 10000  # this is to prevent very long epochs for very large datasets
        self.model.fit(channel.generator_train,
                       steps_per_epoch=min(self.max_steps_per_epoch, len(channel.generator_train)),
                       epochs=self.config.epochs,
                       validation_data=channel.generator_val,
                       callbacks=cbs,
                       verbose=True)

    def save(self):
        """
        Save trained model.
        """

        self.model.save(self.model_path)

    def aggregate_predictions(self, y_hat_batch, method='first'):
        """
        Aggregates predictions for each timestep. When predicting n steps
        ahead where n > 1, will end up with multiple predictions for a
        timestep.

        Args:
            y_hat_batch (arr): predictions shape (<batch length>, <n_preds)
            method (string): indicates how to aggregate for a timestep - "first"
                or "mean"
        """

        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - self.config.prediction_window_size
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, channel, save_y_hat: bool = True):
        """
        Used trained LSTM model to predict test data arriving in batches.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
        """

        num_batches = len(channel.generator_test)
        if num_batches < 1:
            raise ValueError('l_s ({}) too large for stream length.'
                             .format(self.config.window_size))

        # simulate data arriving in batches, predict each batch

        y_hat_batch = self.model.predict(channel.generator_test)
        self.aggregate_predictions(y_hat_batch)

        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))
        
        # Revert standardization
        self.y_hat = self.y_hat * channel.generator_test.train_stds[channel.generator_test.target_channel_index] + \
        channel.generator_test.train_means[channel.generator_test.target_channel_index]

        channel.y_hat = self.y_hat

        if save_y_hat:
            np.save(os.path.join('data', self.run_id, 'y_hat', '{}.npy'
                                 .format(self.chan_id)), self.y_hat)

        return channel

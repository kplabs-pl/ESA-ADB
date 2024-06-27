# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 07:03:05 2022

@author: gastong@fing.edu.uy
"""


from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv1D, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import pandas as pd
import numpy as np
from .metrics import per_event_f_score
from sklearn.metrics import fbeta_score
import os
import pickle
from tqdm import tqdm


@keras.utils.register_keras_serializable()
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, name=None, k=1, **kwargs):
        super(Sampling, self).__init__(name=name)
        self.k = k
        super(Sampling, self).__init__(**kwargs)

    def get_config(self):
        config = super(Sampling, self).get_config()
        config['k'] = self.k
        return config #dict(list(config.items()))

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        seq = K.shape(z_mean)[1]
        dim = K.shape(z_mean)[2]
        epsilon = K.random_normal(shape=(batch, seq, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class DCVAE:

    def __init__(self,
                 T=32,
                 M=12,
                 target_channels_indices=None,
                 cnn_units = [32, 16, 1],
                 dil_rate = [1,8,16],
                 kernel=2,
                 strs=1,
                 batch_size=32,
                 J=1,
                 epochs=100,
                 learning_rate=1e-3,
                 lr_decay=True,
                 decay_rate=0.96,
                 decay_step=1000,
                 name = '',
                 epsilon = 1e-12,
                 summary=True,
                 ):


        # network parameters
        input_shape = (T, M)
        M_output = len(target_channels_indices)
        self.M = M
        self.target_channels_indices = target_channels_indices
        self.M_output = M_output
        self.T = T
        self.J = J
        self.batch_size = batch_size
        self.epochs = epochs
        self.name = name


        # model = encoder + decoder

        # Build encoder model
	        # =============================================================================
        # Input
        inputs = Input(shape=input_shape, name='input')
        outputs = Input(shape=(T, M_output), name="output")

        # Hidden layers (1D Dilated Convolution)
        # First
        h_enc_cnn = Conv1D(cnn_units[0], kernel, activation='tanh', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001),
                           strides=strs, padding="causal",
                           dilation_rate=dil_rate[0], name='cnn_%d'%0)(inputs)
        h_enc_cnn = BatchNormalization()(h_enc_cnn)

        # Middle
        for i in range(len(cnn_units)-2):
            h_enc_cnn = Conv1D(cnn_units[i+1], kernel, activation='tanh', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001),
                           strides=strs, padding="causal",
                           dilation_rate=dil_rate[i+1], name='cnn_%d'%(i+1))(h_enc_cnn)
            h_enc_cnn = BatchNormalization()(h_enc_cnn)

        # Lastest
        z_mean = Conv1D(J, kernel, activation=None,
                           strides=strs, padding="causal",
                           dilation_rate=dil_rate[i+1], name='z_mean')(h_enc_cnn)
        z_log_var = Conv1D(J, kernel, activation=None,
                           strides=strs, padding="causal",
                           dilation_rate=dil_rate[i+1], name='z_log_var')(h_enc_cnn)

        # Reparameterization trick
        # Output
        z = Sampling(name='z')((z_mean, z_log_var))
        # Instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        if summary:
            self.encoder.summary()
        # =============================================================================

        # Build decoder model
        # =============================================================================
        # Input
        latent_inputs = Input(shape=(T, J), name='z_sampling')

        # Hidden layers (1D Dilated Convolution)
        # First
        h_dec_cnn = Conv1D(cnn_units[-1], kernel, activation='elu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001),
                           strides=strs, padding="causal",
                           dilation_rate=dil_rate[-1], name='cnn_-1')(latent_inputs)
        h_dec_cnn = BatchNormalization()(h_dec_cnn)

        #Middle
        for i in range(-2, -len(cnn_units), -1):
            h_dec_cnn = Conv1D(cnn_units[i], kernel, activation='elu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001),
                           strides=strs, padding="causal",
                           dilation_rate=dil_rate[i], name='cnn_%d'%i)(h_dec_cnn)
            h_dec_cnn = BatchNormalization()(h_dec_cnn)

        # Lastest/Output
        x__mean = Conv1D(M_output, kernel, activation=None,
                                  padding="causal",
                                  dilation_rate=dil_rate[0],
                                  name='x__mean_output')(h_dec_cnn)
        x_log_var = Conv1D(M_output, kernel, activation=None,
                                  padding="causal",
                                  dilation_rate=dil_rate[0],
                                  name='x_log_var_output')(h_dec_cnn)

        # Instantiate decoder model
        self.decoder = Model(latent_inputs, [x__mean, x_log_var], name='decoder')
        if summary:
            self.decoder.summary()
        # =============================================================================

        # Instantiate DC-VAE model
        # =============================================================================
        [x__mean, x_log_var] = self.decoder(self.encoder(inputs)[2])
        self.vae = Model([inputs, outputs], [x__mean, x_log_var], name='vae')

        # Loss
        # Reconstruction term
        MSE = -0.5*K.mean(K.square((outputs - x__mean)/K.exp(x_log_var)),axis=-1) #Mean in M
        sigma_trace = -K.mean(x_log_var, axis=(-1)) #Mean in M
        log_likelihood = MSE+sigma_trace
        reconstruction_loss = K.mean(-log_likelihood) #Mean in the batch and T

        # Priori hypothesis term
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.mean(kl_loss, axis=-1) #Mean in J
        kl_loss *= -0.5
        kl_loss = tf.reduce_mean(kl_loss) #Mean in the batch and T

        # Total
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)

        # Learning rate
        if lr_decay:
            lr = optimizers.schedules.ExponentialDecay(learning_rate,
                                                    decay_steps=decay_step,
                                                    decay_rate=decay_rate,
                                                    staircase=True,
                                                    )
        else:
            lr = learning_rate

        # Optimaizer
        opt = optimizers.Adam(learning_rate=lr)

        # Metrics
        self.vae.add_metric(reconstruction_loss, name='reconst')
        self.vae.add_metric(kl_loss, name='kl')


        self.vae.compile(optimizer=opt)


    def fit(self, channel, model_path):

        # Callbacks
        early_stopping_cb = keras.callbacks.EarlyStopping(min_delta=1e-3,
                                                      patience=20,
                                                      verbose=1,
                                                      mode='min')
        model_checkpoint_cb= keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            verbose=1,
            mode='min',
            save_best_only=True)


        # Model train
        self.max_steps_per_epoch = 1000
        self.history_ = self.vae.fit(channel.generator_train,
                     epochs=self.epochs,
                     steps_per_epoch=min(self.max_steps_per_epoch, len(channel.generator_train)),
                     validation_data = channel.generator_val,
                     callbacks=[early_stopping_cb,
                                model_checkpoint_cb],
                     verbose=True
                     )
        pd.DataFrame.from_dict(self.history_.history).to_csv(model_path + '_history.csv', index=False)

        # Plot loss curves
        plt.plot(self.history_.history["loss"], label="Training Loss")
        plt.plot(self.history_.history["val_loss"], label="Validation Loss")
        plt.plot(self.history_.history["reconst"], label="Training Reconstruction")
        plt.plot(self.history_.history["val_reconst"], label="Validation Reconstruction")
        plt.plot(self.history_.history["kl"], label="Training KL")
        plt.plot(self.history_.history["val_kl"], label="Validation KL")

        plt.legend()
        plt.savefig(model_path + '_loss.jpg')

        # Save models
        self.encoder.save(model_path+'_encoder.h5')
        self.decoder.save(model_path+'_decoder.h5')
        self.vae.save(model_path+'_complete.h5')

        return self


    def alpha_selection(self, X, y, model_path, channel, load_model=False, custom_metrics=False):

        # Model
        if load_model:
            self.vae = keras.models.load_model(model_path,
                                                  custom_objects={'sampling': Sampling},
                                                  compile = False)

        # Inference model. Auxiliary model so that in the inference
        # the prediction is only the last value of the sequence
        inp = Input(shape=(self.T, self.M))
        output = Input(shape=(self.T, self.M_output))
        x = self.vae([inp, output]) # apply trained model on the input
        out = Lambda(lambda y: [y[0][:,-1,:], y[1][:,-1,:]])(x)
        inference_model = Model([inp, output], out)

        # Data
        dataset_val_th = timeseries_dataset_from_array(
            X[..., channel.config.input_channel_indices], None, self.T, sequence_stride=1, sampling_rate=1,
            batch_size=self.batch_size)

        # Predict
        reconstruct = []
        sig = []
        for batch in dataset_val_th:
            prediction = inference_model([batch, batch[..., :self.M_output]])
            reconstruct.append(prediction[0].numpy())
            sig.append(prediction[1].numpy())

        reconstruct = np.concatenate(reconstruct)
        sig = np.sqrt(np.exp(np.concatenate(sig)))

        # Data evaluate (The first T-1 data are discarded)
        X_evaluate = X[..., channel.config.target_channel_indices][self.T - 1:]
        y_evaluate = y[..., channel.config.target_channel_indices][self.T - 1:]

        print('Alpha selection...')
        best_f1 = np.zeros(self.M_output)
        max_alpha = 7
        best_alpha_up = max_alpha * np.ones(self.M_output)
        best_alpha_down = max_alpha * np.ones(self.M_output)
        from time import time
        start = time()
        for alpha_up in np.arange(max_alpha, 1, -1):
            for alpha_down in np.arange(max_alpha, 1, -1):

                pre_predict = (X_evaluate < reconstruct - alpha_down * sig) | (
                            X_evaluate > reconstruct + alpha_up * sig)
                pre_predict = pre_predict.astype(int)

                for c in range(self.M_output):
                    if custom_metrics:
                        f1_value = per_event_f_score(y_evaluate[:, c], pre_predict[:, c], beta=0.5)
                    else:
                        f1_value = fbeta_score(y_evaluate[:, c], pre_predict[:, c], beta=1.0)

                    if f1_value >= best_f1[c]:
                        best_f1[c] = f1_value
                        best_alpha_up[c] = alpha_up
                        best_alpha_down[c] = alpha_down
            print(alpha_up, time() - start)
        print("optimized", time() - start)
        print(best_alpha_up)
        print(best_alpha_down)
        self.alpha_up = best_alpha_up
        self.alpha_down = best_alpha_down
        self.f1_val = best_f1

        with open(model_path + '_alpha_up.pkl', 'wb') as f:
            pickle.dump(best_alpha_up, f)
            f.close()
        with open(model_path + '_alpha_down.pkl', 'wb') as f:
            pickle.dump(best_alpha_down, f)
            f.close()

        return self

    def predict(self, args, channel,
                load_model=False,
                load_alpha=True,
                alpha_set_up=[],
                alpha_set_down=[]):


        # Predictions
        reconstruction_path = args.modelInput + ".reconstruction.csv"
        deviation_path = args.modelInput + ".deviation.csv"
        if os.path.isfile(reconstruction_path) and os.path.isfile(deviation_path): # avoid prediction if reconstructions are already available
            reconstruct = np.loadtxt(reconstruction_path, delimiter=",", skiprows=self.T)
            sig = np.loadtxt(deviation_path, delimiter=",", skiprows=self.T)
        else:
            # Trained model
            if load_model:
                self.vae = keras.models.load_model(args.modelInput,
                                                   custom_objects={'sampling': Sampling},
                                                   compile=False)

            # Inference model. Auxiliary model so that in the inference
            # the prediction is only the last value of the sequence
            inp = Input(shape=(self.T, self.M))
            output = Input(shape=(self.T, self.M_output))
            x = self.vae([inp, output])  # apply trained model on the input
            out = Lambda(lambda y: [y[0][:, -1, :], y[1][:, -1, :]])(x)
            inference_model = Model([inp, output], out)

            reconstruct = []
            sig = []
            for i in tqdm(range(len(channel.generator_test))):
                prediction = inference_model(channel.generator_test[i])
                reconstruct.append(prediction[0].numpy())
                sig.append(np.sqrt(np.exp(prediction[1].numpy())))

            reconstruct = np.concatenate(reconstruct)
            sig = np.concatenate(sig)

            # Revert standardization
            reconstruct = reconstruct * channel.generator_test.train_stds[channel.config.target_channel_indices] + channel.generator_test.train_means[channel.config.target_channel_indices]
            sig *= channel.generator_test.train_stds[channel.config.target_channel_indices]

            first_row = reconstruct[0]
            np.savetxt(reconstruction_path, np.concatenate(([first_row] * (self.T - 1), reconstruct)),
                       delimiter=",", header=",".join(channel.id), comments="")

            first_row = sig[0]
            np.savetxt(deviation_path, np.concatenate(([first_row] * (self.T - 1), sig)),
                       delimiter=",", header=",".join(channel.id), comments="")

        # Thresholds
        if len(alpha_set_up) == self.M_output:
            alpha_up = np.array(alpha_set_up)
        elif load_alpha:
            with open(args.modelInput + '_alpha_up.pkl', 'rb') as f:
                alpha_up = pickle.load(f)
        else:
            alpha_up = self.alpha_up

        if len(alpha_set_down) == self.M_output:
            alpha_down = np.array(alpha_set_down)
        elif load_alpha:
            with open(args.modelInput + '_alpha_down.pkl', 'rb') as f:
                alpha_down = pickle.load(f)
        else:
            alpha_down = self.alpha_down

        thdown = reconstruct - alpha_down*sig
        thup = reconstruct + alpha_up*sig

        # Evaluation
        X_evaluate = channel.generator_test.data[0, self.T - 1:][..., channel.config.target_channel_indices]
        pred = (X_evaluate < thdown) | (X_evaluate > thup)
        np.savetxt(args.dataOutput, np.concatenate((np.zeros((self.T - 1, self.M_output)), pred)).astype(np.uint8), delimiter=",")


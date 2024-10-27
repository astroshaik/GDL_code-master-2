import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Flatten, Dense, Conv2DTranspose, 
    Reshape, Activation, BatchNormalization, 
    LeakyReLU, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.utils import plot_model
import numpy as np
import os
import pickle

# Define CustomCallback
class CustomCallback(Callback):
    def __init__(self, run_folder, print_every_n_batches, initial_epoch, autoencoder):
        super().__init__()
        self.run_folder = run_folder
        self.print_every_n_batches = print_every_n_batches
        self.initial_epoch = initial_epoch
        self.autoencoder = autoencoder

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.print_every_n_batches == 0:
            print(f"Epoch {epoch + self.initial_epoch} - loss: {logs['loss']:.4f}")

# Define step_decay_schedule function
def step_decay_schedule(initial_lr, decay_factor, step_size):
    def schedule(epoch):
        return initial_lr * (decay_factor ** (epoch // step_size))
    return tf.keras.callbacks.LearningRateScheduler(schedule)

class Autoencoder:
    def __init__(self, input_dim, encoder_conv_filters, encoder_conv_kernel_size,
                 encoder_conv_strides, decoder_conv_t_filters, decoder_conv_t_kernel_size,
                 decoder_conv_t_strides, z_dim, use_batch_norm=False, use_dropout=False):
        
        self.name = 'autoencoder'
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        # Ensure all lists have the same length
        if not (len(encoder_conv_filters) == len(encoder_conv_kernel_size) == len(encoder_conv_strides)):
            raise ValueError("Encoder filter sizes, kernel sizes, and strides must have the same length.")
        
        if not (len(decoder_conv_t_filters) == len(decoder_conv_t_kernel_size) == len(decoder_conv_t_strides)):
            raise ValueError("Decoder filter sizes, kernel sizes, and strides must have the same length.")

        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)

        self._build()

    def _build(self):
        ### THE ENCODER
        encoder_input = tf.keras.Input(shape=self.input_dim, name='encoder_input')
        x = encoder_input

        for i in range(self.n_layers_encoder):
            conv_layer = Conv2D(
                filters=self.encoder_conv_filters[i],
                kernel_size=self.encoder_conv_kernel_size[i],
                strides=self.encoder_conv_strides[i],
                padding='same',
                name='encoder_conv_' + str(i)
            )

            x = conv_layer(x)
            x = LeakyReLU()(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        shape_before_flattening = tf.keras.backend.int_shape(x)[1:]        
        x = Flatten()(x)
        encoder_output = Dense(self.z_dim, name='encoder_output')(x)

        self.encoder = Model(encoder_input, encoder_output)

        ### THE DECODER
        decoder_input = tf.keras.Input(shape=(self.z_dim,), name='decoder_input')
        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(
                filters=self.decoder_conv_t_filters[i],
                kernel_size=self.decoder_conv_t_kernel_size[i],
                strides=self.decoder_conv_t_strides[i],
                padding='same',
                name='decoder_conv_t_' + str(i)
            )

            x = conv_t_layer(x)

            if i < self.n_layers_decoder - 1:
                x = LeakyReLU()(x)

                if self.use_batch_norm:
                    x = BatchNormalization()(x)

                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation('sigmoid')(x)

        decoder_output = x
        self.decoder = Model(decoder_input, decoder_output)

        ### THE FULL AUTOENCODER
        model_input = encoder_input
        model_output = self.decoder(encoder_output)
        self.model = Model(model_input, model_output)

        # Print model summary for verification
        print("Autoencoder model summary:")
        self.model.summary()

    def compile(self, learning_rate):
        self.learning_rate = learning_rate
        optimizer = Adam(learning_rate=learning_rate)

        def r_loss(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])  # Use tf.reduce_mean

        self.model.compile(optimizer=optimizer, loss=r_loss)

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim,
                self.encoder_conv_filters,
                self.encoder_conv_kernel_size,
                self.encoder_conv_strides,
                self.decoder_conv_t_filters,
                self.decoder_conv_t_kernel_size,
                self.decoder_conv_t_strides,
                self.z_dim,
                self.use_batch_norm,
                self.use_dropout,
            ], f)

        self.plot_model(folder)

    def load_weights(self, filepath):
        # Print model summary before loading weights
        print("Model summary before loading weights:")
        self.model.summary()
        self.model.load_weights(filepath)

        # Print model summary after loading weights
        print("Model summary after loading weights:")
        self.model.summary()

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=100,
              initial_epoch=0, lr_decay=1):

        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        # Ensure the filepath ends with .weights.h5
        checkpoint_path = os.path.join(run_folder, 'weights/weights.h5')
        checkpoint2 = ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

        callbacks_list = [checkpoint2, custom_callback, lr_sched]

        self.model.fit(
            x_train,
            x_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks_list
        )

    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder, 'viz/model.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.encoder, to_file=os.path.join(run_folder, 'viz/encoder.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.decoder, to_file=os.path.join(run_folder, 'viz/decoder.png'), show_shapes=True, show_layer_names=True)

# Note: Ensure that you have TensorFlow installed in your environment to run this code.

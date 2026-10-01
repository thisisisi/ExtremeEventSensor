#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benedikt Barthel Sorensen <bbarthel@mit.edu>
"""


import tensorflow as tf
import keras


import tensorflow.keras.backend as kb
#from keras.layers.core import Flatten, Dense, Dropout, Lambda

from tensorflow.keras.utils import register_keras_serializable


from loss_funs_qg import param_loss_2D, vae_loss_2D

@register_keras_serializable(package="custom", name="sampling")
def sampling(args, latent_dim=None):
    z_mean, z_sigma = args
    shp = (tf.shape(z_mean)[0], tf.shape(z_mean)[1], latent_dim)
    eps = kb.random_normal(shape=shp, mean=0., stddev=1.)
    return z_mean + z_sigma * eps

def get_model(model_type,used_features,loss_data, lr = 1e-3, **kwargs):

    out_features = kwargs.get("out_features", None)
    inter_dim    = kwargs.get("inter_dim", 60)
    latent_dim   = kwargs.get("latent_dim", 60)

    if model_type == 'LSTM':
        print("Standard LSTM Network")
        network = get_lstm(used_features,loss_data, lr=lr, out_features=out_features)

    if model_type == 'VAE-RNN':
        print("VAE LSTM Network")
        network = get_vaernn(used_features,loss_data)

    if model_type == 'STORN':
        print("STORN LSTM Network")
        network = get_storn(used_features,loss_data, lr=lr, out_features=out_features, inter_dim=inter_dim, latent_dim=latent_dim)

    if model_type == 'VRNN':
         print("VRNN-I LSTM Network")
         network = get_vrnn(used_features,loss_data)


    network.summary()
    return network


def get_lstm(used_features,loss_data, lr = 1e-3, out_features = None, **kwargs):

    if out_features is None:
        out_features = used_features

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(None,used_features), name='field_in'))
    model.add(tf.keras.layers.Dense(60, activation='tanh'))

    model.add(tf.keras.layers.LSTM(input_shape=(None,60),units=60, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.00, recurrent_dropout=0.00, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False))

    model.add(tf.keras.layers.Dense(out_features, activation='linear'))
    model.compile(loss=param_loss_2D(alpha=loss_data), optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    return model


# VAE-RNN with lambda =  10^-4 loss
def get_vaernn(used_features,loss_data):

    features = used_features
    inter_dim = 60
    latent_dim = 60

    def sampling(args):
        z_mean, z_sigma = args
        batch_size = tf.shape(z_mean)[0] # <================
        time_hist = tf.shape(z_mean)[1] # <================
        epsilon = kb.random_normal(shape=(batch_size,time_hist, latent_dim), mean=0., stddev=1.)
        return z_mean + (z_sigma)*epsilon


    # timesteps, features
    x_in = tf.keras.layers.Input(shape= (None, features),name = 'Input')

    # Dense Layer
    h = tf.keras.layers.Dense(inter_dim, activation='tanh', name="Encoder1")(x_in)

   # Pre Z layer LSTM
    #h = tf.keras.layers.LSTM(input_shape=(None,inter_dim),units=60, activation='relu')(h)
    h = tf.keras.layers.LSTM(input_shape=(None,60),units=60, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.00, recurrent_dropout=0.00, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False, name="LSTM1")(h)

    #z_layer
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean", activation='linear')(h)
    z_sigma = tf.keras.layers.Dense(latent_dim, name="z_var", activation='softplus',kernel_initializer='zeros')(h)
    z = tf.keras.layers.Lambda(sampling, name="Sample")([z_mean, z_sigma])


    # Reconstruction decoder
    y_pred = tf.keras.layers.Dense(used_features, activation='linear', name="Decoder1")(z)

    # Concatonate
    output = tf.keras.layers.Concatenate(axis=-1, name="Concatenate")([y_pred, z_mean, z_sigma])

    # Create Model
    model = tf.keras.Model(inputs=[x_in],outputs=[output],)
    model.compile(loss=vae_loss_2D(alpha=loss_data), optimizer='adam' )


    return model


# STORN with lambda =  10^-4 loss

def get_storn(used_features, loss_data, lr=1e-3, out_features=None,
              inter_dim=60, latent_dim=60, **kwargs):
    """
    Output = concat([y_pred, z_mean, z_sigma]) where:
      - y_pred has dimension out_features (defaults to used_features)
      - z_mean, z_sigma have dimension latent_dim each
    """
    if out_features is None:
        out_features = used_features

    x_in = tf.keras.layers.Input(shape=(None, used_features), name='Input')

    h = tf.keras.layers.Dense(inter_dim, activation='tanh', name='Encoder1')(x_in)
    z_mean  = tf.keras.layers.Dense(latent_dim, activation='linear',   name='z_mean')(h)
    z_sigma = tf.keras.layers.Dense(latent_dim, activation='softplus', name='z_var',
                                    kernel_initializer='zeros')(h)

    # uses the @register_keras_serializable sampling() defined above
    #z = tf.keras.layers.Lambda(sampling, name='Sample')([z_mean, z_sigma])
    z = tf.keras.layers.Lambda(lambda x: sampling(x, latent_dim))( [z_mean, z_sigma] )


    phi = tf.keras.layers.Concatenate(axis=-1, name='Concatenate1')([h, z])
    d   = tf.keras.layers.LSTM(input_shape=(None,latent_dim+inter_dim),units=inter_dim, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.00, recurrent_dropout=0.00, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False, name="LSTM2")(phi)

    y_pred = tf.keras.layers.Dense(out_features, activation='linear', name='Decoder1')(d)
    output = tf.keras.layers.Concatenate(axis=-1, name='Concatenate2')([y_pred, z_mean, z_sigma])

    model = tf.keras.Model(inputs=x_in, outputs=output)
    model.compile(loss=vae_loss_2D(alpha=loss_data),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr), weighted_metrics=[])

    return model


# VRNN-I with lambda =  10^-4 loss
def get_vrnn(used_features,loss_data):

    features = used_features
    inter_dim = 60
    latent_dim = 60

    def sampling(args):
        z_mean, z_sigma = args
        batch_size = tf.shape(z_mean)[0] # <================
        time_hist = tf.shape(z_mean)[1] # <================
        epsilon = kb.random_normal(shape=(batch_size,time_hist, latent_dim), mean=0., stddev=1.)
        return z_mean + (z_sigma)*epsilon


    # timesteps, features
    x_in = tf.keras.layers.Input(shape= (None, features),name = 'Input')

    # Dense Layer
    h = tf.keras.layers.Dense(inter_dim, activation='tanh', name="Encoder1")(x_in)


    #z_layer
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean", activation='linear')(h)
    z_sigma = tf.keras.layers.Dense(latent_dim, name="z_var", activation='softplus',kernel_initializer='zeros')(h)
    z = tf.keras.layers.Lambda(sampling, name="Sample")([z_mean, z_sigma])

    # Concatonate
    phi =  tf.keras.layers.Concatenate(axis=-1, name="Concatenate1")([h,z])

    # Post Z layer LSTM
    d = tf.keras.layers.LSTM(input_shape=(None,latent_dim+inter_dim),units=inter_dim, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.00, recurrent_dropout=0.00, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False, name="LSTM2")(phi)

    # Concatonate
    zd =  tf.keras.layers.Concatenate(axis=-1, name="Concatenate0")([z,d])


    # Reconstruction decoder
    y_pred = tf.keras.layers.Dense(used_features, activation='linear', name="Decoder1")(zd)

    # Concatonate
    output = tf.keras.layers.Concatenate(axis=-1, name="Concatenate2")([y_pred, z_mean, z_sigma])

    # Create Model
    model = tf.keras.Model(inputs=[x_in],outputs=[output],)
    model.compile(loss=vae_loss_2D(alpha=loss_data), optimizer='adam' )


    return model

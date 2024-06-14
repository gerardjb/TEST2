#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Alternative models for CASCADE:

  default_model():
    defines the architecture of Cascade's original deep network

  Created in June 2024

"""

# Try binary cross entropy and just merge into straight prediction of spike time
# Try a RNN, LSTM, GNU
# Outputs of model should be a signmoid should be a sigmoid

from tensorflow.keras.layers import LSTM, Dense, Flatten, MaxPooling1D, Conv1D, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adagrad

def choose_model(model_name):
    if model_name == "default":
      return default_model
    elif model_name == "cRNN":
        return cRNN_no_dense
    elif model_name == "cRNN_no_relu":
        return cRNN_no_relu
        

def default_model(filter_sizes,filter_numbers,dense_expansion,windowsize,loss_function,optimizer):
    inputs = Input(shape=(windowsize,1))

    conv_filter = Conv1D

    outX = conv_filter(filter_numbers[0], filter_sizes[0], strides=1, activation='relu')(inputs)
    outX = conv_filter(filter_numbers[1], filter_sizes[1], activation='relu')(outX)
    outX = MaxPooling1D(2)(outX)
    outX = conv_filter(filter_numbers[2], filter_sizes[2], activation='relu')(outX)
    outX = MaxPooling1D(2)(outX)

    outX = Dense(dense_expansion, activation='relu')(outX) # 'linear' units work here as well!
    outX = Flatten()(outX)
    predictions = Dense(1,activation='linear')(outX) 
    model = Model(inputs=[inputs],outputs=predictions)
    optimizer = Adagrad(learning_rate=0.05)
    model.compile(loss=loss_function, optimizer=optimizer)

    return model

def cRNN(filter_sizes,filter_numbers,dense_expansion,windowsize,loss_function,optimizer):
    inputs = Input(shape=(windowsize,1))

    conv_filter = Conv1D

    outX = conv_filter(filter_numbers[0], filter_sizes[0], strides=1, activation='relu')(inputs)
    outX = conv_filter(filter_numbers[1], filter_sizes[1], activation='relu')(outX)
    outX = MaxPooling1D(2)(outX)
    outX = conv_filter(filter_numbers[2], filter_sizes[2], activation='relu')(outX)
    outX = MaxPooling1D(2)(outX)

    # Try LSTM before or instead of first dense layer
    outX = LSTM(dense_expansion)(outX)
    outX = Dense(dense_expansion, activation='relu')(outX) # 'linear' units work here as well!
    outX = Flatten()(outX)
    predictions = Dense(1,activation='linear')(outX) 
    model = Model(inputs=[inputs],outputs=predictions)
    optimizer = Adagrad(learning_rate=0.05)
    model.compile(loss=loss_function, optimizer=optimizer)

    return model

def cRNN_no_dense(filter_sizes,filter_numbers,dense_expansion,windowsize,loss_function,optimizer):
    inputs = Input(shape=(windowsize,1))

    conv_filter = Conv1D

    outX = conv_filter(filter_numbers[0], filter_sizes[0], strides=1, activation='relu')(inputs)
    outX = conv_filter(filter_numbers[1], filter_sizes[1], activation='relu')(outX)
    outX = MaxPooling1D(2)(outX)
    outX = conv_filter(filter_numbers[2], filter_sizes[2], activation='relu')(outX)
    outX = MaxPooling1D(2)(outX)

    outX = LSTM(dense_expansion)(outX)
    outX = Flatten()(outX)
    predictions = Dense(1,activation='linear')(outX) 
    model = Model(inputs=[inputs],outputs=predictions)
    optimizer = Adagrad(learning_rate=0.05)
    model.compile(loss=loss_function, optimizer=optimizer)

    return model

# Use Tensorboard 
def cRNN_sigmoid(filter_sizes,filter_numbers,dense_expansion,windowsize,loss_function,optimizer):
    inputs = Input(shape=(windowsize,1))

    conv_filter = Conv1D

    # Add maxpool layer after conv
    # Can increase number of filters
    # Add batch normalization
    outX = conv_filter(filter_numbers[0], filter_sizes[0], strides=1, activation='relu')(inputs)
    outX = conv_filter(filter_numbers[1], filter_sizes[1], activation='relu')(outX)
    outX = MaxPooling1D(2)(outX)
    outX = conv_filter(filter_numbers[2], filter_sizes[2], activation='relu')(outX)
    outX = MaxPooling1D(2)(outX)

    outX = LSTM(dense_expansion)(outX)
    outX = Flatten()(outX)
    # Change activation function to sigmoid
    predictions = Dense(1,activation='sigmoid')(outX) 
    model = Model(inputs=[inputs],outputs=predictions)
    optimizer = Adagrad(learning_rate=0.05)
    # Change loss function to binary cross entropy
    model.compile(loss=loss_function, optimizer=optimizer)

    return model

# To try
# ReLU with LSTM
# Make Y binary spike times

def cRNN_no_relu(filter_sizes,filter_numbers,dense_expansion,windowsize,loss_function,optimizer):
    inputs = Input(shape=(windowsize,1))

    #conv_filter = Conv1D

    # Add maxpool layer after conv
    # Can increase number of filters
    # Add batch normalization
    outX = Conv1D(filter_numbers[0], filter_sizes[0], strides=1, activation='relu')(inputs)
    outX = Conv1D(filter_numbers[1], filter_sizes[1], activation='relu')(outX)
    outX = MaxPooling1D(2)(outX)
    outX = Conv1D(filter_numbers[2], filter_sizes[2], activation='relu')(outX)
    outX = MaxPooling1D(2)(outX)

    outX = LSTM(dense_expansion)(outX)
    outX = Flatten()(outX)
    # Change activation function to sigmoid
    predictions = Dense(1,activation='relu')(outX) 
    model = Model(inputs=[inputs],outputs=predictions)
    optimizer = Adagrad(learning_rate=0.05)
    # Change loss function to binary cross entropy
    model.compile(loss=loss_function, optimizer=optimizer)

    return model

# Try w/ and w/out relu for LSTM
# Try w/ and w/out smoothing




# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:53:24 2018

@author: Sandiagal
"""

# GRADED FUNCTION: HappyModel

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import regularizers
from keras.applications import mobilenet
from keras.layers.advanced_activations import LeakyReLU

def YannLeCun(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=(9, 9),
                     strides=1, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=8, kernel_size=(7, 7),
                     strides=1, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=16, kernel_size=(5, 5),
                     strides=1, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     strides=1, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    # the model so far outputs 3D feature maps (height, width, features)

    # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())
    model.add(Dense(units=32, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=16, activation='sigmoid',
                    kernel_regularizer=regularizers.l2(0.01)))

    return model

def mobilenet2(input_shape):
    base_model = mobilenet.MobileNet(
        input_shape=input_shape, dropout=0.1, include_top=False, weights='imagenet', pooling='max')

    output = base_model.output
    output = Dense(units=256, activation='relu',
                    kernel_regularizer=regularizers.l2(0.1))(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)

    output = Dense(units=128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.1))(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)

    predictions = Dense(units=16, activation='softmax',
                    kernel_regularizer=regularizers.l2(0.1))(output)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def mobilenetFC(input_shape, classes):
    base_model = mobilenet.MobileNet(alpha=0.5,
        input_shape=input_shape, dropout=0.1, include_top=False, weights='imagenet', pooling=None)

    output = base_model.output
    output = MaxPooling2D(pool_size=(7, 7))(output)
    output = Conv2D(filters=256, kernel_size=(1, 1),
                     strides=1, activation='relu', kernel_regularizer=regularizers.l2(0.1))(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)

    output = Conv2D(filters=128, kernel_size=(1, 1),
                     strides=1, activation='relu', kernel_regularizer=regularizers.l2(0.1))(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)


    predictions = Conv2D(filters=classes, kernel_size=(1, 1),
                     strides=1, activation='softmax', kernel_regularizer=regularizers.l2(0.1))(output)
    predictions=GlobalAveragePooling2D()(predictions)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

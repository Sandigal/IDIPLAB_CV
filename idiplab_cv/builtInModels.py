# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:53:24 2018

@author: Sandiagal
"""

# GRADED FUNCTION: HappyModel

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Reshape
from keras import regularizers
from keras.applications import mobilenet


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


def mobilenetGAP(input_shape, classes, dropout=None):
    #    base_model = mobilenet.MobileNet(
    #        input_shape=input_shape, dropout=dropout, include_top=False, weights="imagenet", pooling="avg")
    #    output = base_model.output

#    base_conv = mobilenet.MobileNet(
#        input_shape=(224, 224, 3), dropout=dropout, include_top=False, weights="imagenet", pooling="avg")
    base_model = mobilenet.MobileNet(
        input_shape=input_shape, dropout=dropout, include_top=False, weights=None, pooling="avg")
#    for new_layer, layer in zip(base_model.layers[0:], base_conv.layers[0:]):
#        new_layer.set_weights(layer.get_weights())
    output = base_model.output

    output = Reshape((1, 1, 1024), name='reshape_1')(output)
#    output = Dropout(dropout, name='dropout')(output)
    output = Conv2D(filters=classes, kernel_size=(1, 1),
                    kernel_regularizer=regularizers.l2(1),
                    padding='same', name='conv_preds')(output)
#    output = Dense(units=classes,
#                    kernel_regularizer=regularizers.l2(1),
#                     name='last_preds')(output)
    output = BatchNormalization(name='bn')(output)
    output = Activation('softmax', name='softmax')(output)
#    output = Reshape((classes,), name='reshape')(output)
    output = GlobalAveragePooling2D(name='reshape')(output)
    model = Model(inputs=base_model.input, outputs=output)
    return model


def FC1(input_shape, classes):
    model = Sequential(input_shape=input_shape)
    model.add(MaxPooling2D(pool_size=(7, 7)))
    model.add(Conv2D(filters=256, kernel_size=(1, 1),
                     strides=1, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
#    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=128, kernel_size=(1, 1),
                     strides=1, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
#    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=classes, kernel_size=(1, 1),
                     strides=1, activation='softmax', kernel_regularizer=regularizers.l2(0.1)))
    model.add(GlobalAveragePooling2D())
    return model

#%%

input_shape = (None, None, 3)  # 32的倍数
lenn = 3
model = mobilenetGAP(input_shape, lenn)
model.compile(optimizer="adam", loss='categorical_crossentropy',
              metrics=['accuracy'])
# plot_model(model, to_file='5.4.png', show_shapes=True)
model.summary()


from PIL import Image
import numpy as np
import os
from keras.utils import to_categorical

imgs = np.array([np.array(Image.open("../data/dataset 672x448/crop/P3/" + fname))
                 for fname in os.listdir("../data/dataset 672x448/crop/P3/")])
lable = [0, 0, 1, 1, 2, 2]
lable = to_categorical(lable, lenn)
model.fit(imgs, lable)
preResult = model.predict(imgs)

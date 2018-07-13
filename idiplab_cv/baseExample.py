# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:45:02 2018

@author: Sandiagal
"""

import idiplab_cv.dataset_io as io
from idiplab_cv.builtInModels import mobilenetGAP
from idiplab_cv.preprocess import Imgaug

import numpy as np
from pickle import dump
from time import strftime, localtime
from sklearn.utils import shuffle

from imgaug import augmenters as iaa

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import keras.optimizers as op
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.applications.mobilenet import preprocess_input

# %%

np.random.seed(seed=0)
tf.set_random_seed(seed=0)
nowTime = strftime("%Y%m%d", localtime())

# %%

dataset = io.Dataset(path="dataset", augment=False)
class_to_index, sample_per_class = dataset.load_data()

_, _, imgs_test, labels_test = dataset.train_test_split(test_shape=0.2)

imgs_train, labels_train, imgs_valid, labels_valid = dataset.cross_split(
        total_splits=3, valid_split=0)

labels_train = io.label_str2index(labels_train, class_to_index)
labels_train = io.to_categorical(labels_train, len(class_to_index))
labels_train = io.label_smooth(labels_train, [0, 5, 11, 16])
imgs_train, labels_train = shuffle(imgs_train, labels_train, random_state=0)

labels_valid = io.label_str2index(labels_valid, class_to_index)
labels_valid = io.to_categorical(labels_valid, len(class_to_index))

# %%

model = mobilenetGAP(
    input_shape=(None, None, 3),
    classes=len(class_to_index),
    dropout=0.5)
#freeze_index = model.layers.index(model.get_layer('conv_pw_6_relu'))
# for layer in model.layers[0:freeze_index]:
#    layer.trainable = False

model.compile(
    optimizer=op.adam(lr=0.001, decay=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.summary()

# %%

seq = iaa.Sequential([

    iaa.Fliplr(0.5),

    iaa.CropAndPad(
        percent=(0, 0.1),
        pad_mode=["constant", "edge"],
        pad_cval=(0)
    ),

    iaa.Sometimes(
        1,
        iaa.OneOf([
            iaa.GaussianBlur((0, 3.0)),
            iaa.AverageBlur(k=(2, 7)),
            iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
        ])
    ),

    iaa.OneOf([
        iaa.AdditiveGaussianNoise(scale=(0.0, 0.05*255)),
        iaa.CoarseDropout(0.05, size_percent=0.15)
    ]),

    iaa.OneOf([
        iaa.Add((-40, 0)),
        iaa.Multiply((1, 1.2))
    ]),

    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        rotate=(-10, 10),
        shear=(-5, 5),
        mode=["constant", "edge"],
        cval=(0)
    )

], random_order=True)  # apply augmenters in random order

preprocessor = Imgaug(seq)

# %%

batch_size = 32

train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    preprocessing_function=preprocessor)

valid_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

imgs_train_origin = np.array(dataset.imgs_origin)[
    dataset.train_index][dataset.train_cross_index]
train_datagen.fit(imgs_train_origin)
valid_datagen.fit(imgs_train_origin)

train_generator = train_datagen.flow(
    imgs_train, labels_train, batch_size=batch_size, shuffle=False)
valid_generator = valid_datagen.flow(
    imgs_valid, labels_valid, batch_size=batch_size, shuffle=False)

del valid_datagen, imgs_train_origin, dataset

# %%

checkpoint = ModelCheckpoint(
    filepath="record_epoch.{epoch:02d}_loss.{val_loss:.2f}_acc.{val_acc:.2f}.h5",
    monitor='val_acc',
    verbose=1,
    save_best_only=True)

reduceLR = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    epsilon=0.001,
    cooldown=0,
    min_lr=0)

earlyStop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=11,
    verbose=1)

callbacks_list = [checkpoint, reduceLR, earlyStop]

# %%

epochs = 1

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=15*len(labels_train) / batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=len(labels_valid) / batch_size,
    callbacks=callbacks_list,
    shuffle=False)

# %%

score_predict = model.predict(
    imgs_valid,
    batch_size=batch_size,
    verbose=1)

# %%

model.save(nowTime+"_model.h5")
result = {
    "mean": train_datagen.mean,
    "std": train_datagen.std,
    "score_predict": score_predict,
    "history": history.history}
f = open(nowTime+"_result.h5", "wb")
dump(result, f, True)
f.close()

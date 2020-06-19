# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:45:02 2018

@author: Sandiagal

训练用代码示例
"""

# USAGE
# python3 train_agmt_steps.py

# %% 导入必要包
import os
from pickle import dump
from time import clock
from time import localtime
from time import strftime

from imgaug import augmenters as ia
from keras.applications.resnet50 import preprocess_input
#from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.models import Model
import keras.optimizers as op
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib
matplotlib.use("Agg")
import numpy as np
import tensorflow as tf

import idiplab_cv.dataset_io as io
from idiplab_cv import metrics
from idiplab_cv import models
#from idiplab_cv import recall
from idiplab_cv.preprocess import Imgaug
#from idiplab_cv.preprocess import PreprocessInput
from idiplab_cv.manifold import tensorboard_embed
#from idiplab_cv.losses import focal_loss

# %% 记录系统初始化

print("[INFO] Basic launching...")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

nowTime = strftime("%Y%m%d", localtime())
np.random.seed(seed=0)
tf.set_random_seed(seed=0)

is_exist = os.path.exists('./Record')
if not is_exist:
    os.makedirs('./Record')

# %% 全局变量

FILE_NAME = "model_test"
DATASET_PATH = "dataset_test.h5"
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 16
EPOCHS = 100
DROPOUT = 0.5
INIT_LR = 1e-4
AUGMENT_AMOUNT = 4
FEATURE_LAYER = "activation_98"
WEIGHT_LAYER = "predictions"
LOG_DIR = "./logs"

# %% 数据处理

start = clock()

dataset = io.Dataset()
class_to_index, sample_per_class = dataset.load_data(
    path=DATASET_PATH)
imgs_train_all, _, imgs_test, labels_test = dataset.train_test_split(
    total_splits=4, test_split=3)
imgs_train, labels_train, imgs_valid, labels_valid, names_valid = dataset.cross_split(
    total_splits=3, valid_split=0)

labels_train = io.label_str2index(labels_train, class_to_index)
labels_train = io.to_categorical(labels_train, len(class_to_index))
labels_train = io.label_smooth(labels_train, [0, 6])
labels_valid = io.label_str2index(labels_valid, class_to_index)
labels_valid = io.to_categorical(labels_valid, len(class_to_index))
labels_test = io.label_str2index(labels_test, class_to_index)
labels_test = io.to_categorical(labels_test, len(class_to_index))

imgs_train = imgs_train.astype('float32')
imgs_valid = imgs_valid.astype('float32')
imgs_test = imgs_test.astype('float32')

normalization_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
#    preprocessing_function=PreprocessInput()
)
normalization_datagen.fit(imgs_train_all)
mean = normalization_datagen.mean
std = normalization_datagen.std

#imgs_train = preprocess_input(imgs_train)
#imgs_valid = preprocess_input(imgs_valid)
#imgs_test = preprocess_input(imgs_test)

train_generator = normalization_datagen.flow(
    imgs_train, labels_train,
    batch_size=BATCH_SIZE,
    shuffle=True)
valid_generator = normalization_datagen.flow(
    imgs_valid, labels_valid,
    batch_size=BATCH_SIZE,
    shuffle=False)
test_generator = normalization_datagen.flow(
    imgs_test, labels_test,
    batch_size=BATCH_SIZE,
    shuffle=False)

# %% 训练反馈

checkpoint = ModelCheckpoint(
    filepath="./Record/record_epoch.{epoch:02d}_loss.{val_loss:.2f}_acc.{val_acc:.2f}.h5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True)

reduceLR = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=10,
    verbose=1,
    epsilon=0,
    cooldown=0,
    min_lr=0)

earlyStop = EarlyStopping(
    monitor='val_loss',
    patience=1024,
    verbose=1,
    restore_best_weights=True)

Tensorboard = TensorBoard(
    log_dir=LOG_DIR,
    histogram_freq=0,
    batch_size=BATCH_SIZE,
    write_graph=False,
    write_grads=False,
    write_images=False,
    embeddings_freq=0,
    embeddings_layer_names=None,
    embeddings_metadata=None,
    embeddings_data=None)

callbacks_list = [reduceLR, earlyStop, Tensorboard]

# %% 数据增强方法

seq = ia.Sometimes(
    AUGMENT_AMOUNT/(AUGMENT_AMOUNT+1),
    ia.Sequential([
        ia.Fliplr(0.5),

        ia.CropAndPad(
            percent=(0, 0.1),
            pad_mode=["constant", "edge"],
            pad_cval=(0)
        ),

        ia.Sometimes(
            1,
            ia.OneOf([
                ia.GaussianBlur((0, 1.0)),
                ia.AverageBlur(k=(1, 4)),
                ia.Sharpen(alpha=(0.0, 0.5), lightness=(0.8, 1.2))
            ])
        ),

        ia.AdditiveGaussianNoise(scale=(0.0, 0.075*255)),

        ia.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-10, 10),
            shear=(-5, 5),
            mode=["constant", "edge"],
            cval=(0)
        )

    ], random_order=True))
preprocessor = Imgaug(seq)

augment_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    brightness_range=[0.2, 1.2],
    channel_shift_range=10,
    preprocessing_function=preprocessor
    )
augment_datagen.fit(imgs_train)

train_augment_generator = augment_datagen.flow(
    imgs_train, labels_train,
    batch_size=BATCH_SIZE,
    shuffle=True)
valid_augment_generator = augment_datagen.flow(
    imgs_valid, labels_valid,
    batch_size=BATCH_SIZE,
    shuffle=False)

# %% 训练阶段

print("[INFO] Loading resnet50GAP...")
model = load_model("./Record/"+FILE_NAME+".h5")
#model = models.VGGGAP(
#    input_shape=INPUT_SHAPE,
#    classes=len(class_to_index),
#    dropout=DROPOUT)
model.summary()
model.compile(
    optimizer=op.adam(
        lr=INIT_LR, decay=INIT_LR/EPOCHS, clipnorm=0.001),
    loss="categorical_crossentropy",
    metrics=['accuracy'])

print("[INFO] training...")
History = model.fit_generator(
    generator=train_augment_generator,
    steps_per_epoch=(AUGMENT_AMOUNT+1)*len(labels_train) / BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=valid_augment_generator,
    validation_steps=(AUGMENT_AMOUNT+1)*len(labels_valid) / BATCH_SIZE,
    callbacks=callbacks_list,
    shuffle=True)

print("[INFO] Restoring model weights from the end of the best epoch")
model.set_weights(earlyStop.best_weights)

# %% 收尾

print("[INFO] Tail end...")
#model.save("./Record/"+nowTime+"_model.h5")
model.save("./Record/"+FILE_NAME+"_FT.h5")

getFeatureMaps = Model(
    inputs=model.input,
    outputs=model.get_layer(FEATURE_LAYER).output)
feature_maps = getFeatureMaps.predict_generator(
    generator=test_generator,
    steps=len(labels_test) / BATCH_SIZE,
    verbose=1)

getScoresPredict = K.function([model.get_layer(index=model.layers.index(
    model.get_layer(FEATURE_LAYER))+1).input], [model.output])
[scores_predict] = getScoresPredict([feature_maps])

weights = model.get_layer(WEIGHT_LAYER).get_weights()[0]

class_to_index = dict(sorted(class_to_index.items(),
                             key=lambda item: item[1], reverse=False))
labels_test = np.argmax(labels_test, axis=1)
result = {
    "class_to_index": class_to_index,
    "history": History.history,
    "labels_test": labels_test,
#    "mean": mean,
    "scores_predict": scores_predict,
#    "std": std,
}
#f = open("./Record/"+nowTime+"_result.h5", "wb")
f = open("./Record/"+FILE_NAME+"_result_FT.h5", "wb")
dump(result, f, True)
f.close()

result = {
    "feature_maps": feature_maps,
    "weights": weights}
#f = open("./Record/"+nowTime+"_feature_maps.h5", "wb")
f = open("./Record/"+FILE_NAME+"_feature_maps_FT.h5", "wb")
dump(result, f, True)
f.close()

end = clock()
title = "Training Loss and Accuracy: %.3fs" % (end-start)
metrics.show_history(history=History.history, title=title)

print("[INFO] Embeddings...")
tensorboard_embed(LOG_DIR, imgs_test, labels_test, scores_predict)

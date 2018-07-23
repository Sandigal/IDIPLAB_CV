# -*- coding: utf-8 -*-
""" 内置模型模块。

该模块 :meth:`builtInModels` 将keras提供的模型进一步整合。首先隐藏了初学者不会使用的 ``input_tensor`` 等选项。其次重新调整了模型结构，以解决原模型中的问题：加载了 ``ImageNet`` 上预训练的权值后无法另外调整输出类别数。

带 `GAP` 的模型意味着使用了全局平均池化 ``GLobalAveragePool2D``。最后一层卷积层后面再加一层全局平均池化层，可以输出个二维 Tensor `(sample, channel)`。通过该手法可以大幅减少全连接层的参数，并且也是使用CAM的前提。

模型一览:
    +------------------------------+----------+----------------+----------------+---------------+--------+
    | 模型                         | 大小     | Top-1 准确率   | Top-5 准确率   | 参数数量      | 深度   |
    +==============================+==========+================+================+===============+========+
    | :meth:`xceptionGAP`          | 88 MB    | 0.790          | 0.945          | 22,910,480    | 126    |
    +------------------------------+----------+----------------+----------------+---------------+--------+
    | :meth:`vgg16GAP`             | 528 MB   | 0.715          | 0.901          | 138,357,544   | 23     |
    +------------------------------+----------+----------------+----------------+---------------+--------+
    | :meth:`vgg19GAP`             | 549 MB   | 0.727          | 0.910          | 143,667,240   | 26     |
    +------------------------------+----------+----------------+----------------+---------------+--------+
    | :meth:`resnet50GAP`          | 99 MB    | 0.759          | 0.929          | 25,636,712    | 168    |
    +------------------------------+----------+----------------+----------------+---------------+--------+
    | :meth:`inceptionV3GAP`       | 92 MB    | 0.788          | 0.944          | 23,851,784    | 159    |
    +------------------------------+----------+----------------+----------------+---------------+--------+
    | :meth:`inceptionResnetV2GAP` | 215 MB   | 0.804          | 0.953          | 55,873,736    | 572    |
    +------------------------------+----------+----------------+----------------+---------------+--------+
    | :meth:`mobilenetGAP`         | 17 MB    | 0.665          | 0.871          | 4,253,864     | 88     |
    +------------------------------+----------+----------------+----------------+---------------+--------+
    | :meth:`denseNet121GAP`       | 33 MB    | 0.745          | 0.918          | 8,062,504     | 121    |
    +------------------------------+----------+----------------+----------------+---------------+--------+
    | :meth:`denseNet169GAP`       | 57 MB    | 0.759          | 0.928          | 14,307,880    | 169    |
    +------------------------------+----------+----------------+----------------+---------------+--------+
    | :meth:`denseNet201GAP`       | 80 MB    | 0.770          | 0.933          | 20,242,984    | 201    |
    +------------------------------+----------+----------------+----------------+---------------+--------+



像进一步了解模型的配置，请阅读keras官方文档中的 `Applications`_ 章节。

.. _Applications: https://keras.io/applications/

"""

# Author: Sandiagal <sandiagal2525@gmail.com>,
# License: GPL-3.0


from keras import applications
from keras import regularizers
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras.models import Model
from keras.models import Sequential


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


def mobilenetGAP(input_shape, classes, dropout=1e-3):
    """在 ImageNet 上预训练的 MobileNet 模型。

    要通过 `load_model` 载入 MobileNet 模型，你需要导入自定义对象 `relu6` 和 `DepthwiseConv2D` 并通过 `custom_objects` 传参。

    下面是示例代码 ::

        model = load_model('mobilenet.h5', custom_objects={
                           'relu6': mobilenet.relu6,
                           'DepthwiseConv2D': mobilenet.DepthwiseConv2D})

    feature_layer = "conv_pw_13_relu"
    weight_layer = "conv_preds"


    """
    base_conv = applications.mobilenet.MobileNet(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg")
    base_model = applications.mobilenet.MobileNet(
        input_shape=input_shape,
        dropout=dropout,
        include_top=False,
        weights=None,
        pooling="avg")
    for new_layer, layer in zip(base_model.layers[0:], base_conv.layers[0:]):
        new_layer.set_weights(layer.get_weights())
    x = base_model.output

    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1),
               padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model


def xceptionGAP(input_shape, classes):
    """
    feature_layer = "block14_sepconv2_act"
    weight_layer = "predictions"
    """

    base_model = applications.xception.Xception(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling="avg")
    x = base_model.output
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def resnet50GAP(input_shape, classes):
    """
    feature_layer = "activation_49"
    weight_layer = "fc1000"


    """
    base_model = applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling="avg")
    x = base_model.output
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def inceptionV3GAP(input_shape, classes):
    """
    feature_layer = "mixed10"
    weight_layer = "predictions"


    """
    base_model = applications.inception_v3.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling="avg")
    x = base_model.output
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def inceptionResnetV2GAP(input_shape, classes):
    """
    feature_layer = "conv_7b_ac"
    weight_layer = "predictions"


    """
    base_model = applications.inception_resnet_v2.InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling="avg")
    x = base_model.output
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model

def denseNetGAP(input_shape, classes, depth):
    """
    feature_layer = "conv5_block16_concat"
    feature_layer = "conv5_block32_concat"
    feature_layer = "conv5_block32_concat"
    weight_layer = "fc1000"
    """

    if depth is 121:
        base_model = applications.densenet.DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling="avg")
    elif depth is 169:
        base_model = applications.densenet.DenseNet169(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling="avg")
    else:
        base_model = applications.densenet.DenseNet201(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling="avg")

    x = base_model.output
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
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

# %%


#input_shape = (324, 224, 3)  # 32的倍数
#lenn = 11
#model = denseNetGAP(input_shape, lenn, 201)
#model.compile(optimizer="adam", loss='categorical_crossentropy',
#              metrics=['accuracy'])
# plot_model(model, to_file='5.4.png', show_shapes=True)
#model.summary()
#
#
#from PIL import Image
#import numpy as np
#import os
#from keras.utils import to_categorical
#
# imgs = np.array([np.array(Image.open("../data/dataset 672x448/crop/P3/" + fname))
#                 for fname in os.listdir("../data/dataset 672x448/crop/P3/")])
#lable = [0, 0, 1, 1, 2, 2]
#lable = to_categorical(lable, lenn)
#model.fit(imgs, lable)
#preResult = model.predict(imgs)

# -*- coding: utf-8 -*-
"""
该模块 :meth:`models` 包含内置模型的类和函数。

Note:
        请注意 :meth:`models` 中的 `input_shape` 的格式为 `(height, width, channel)`。
"""

# Author: Sandiagal <sandiagal2525@gmail.com>,
# License: GPL-3.0


from keras import applications
from keras import regularizers
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import concatenate
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
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


def xceptionGAP(input_shape, classes, dropout=1e-3, include_top=True):
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

    if include_top is True:
        x = Dropout(dropout, name='dropout')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def VGGGAP(input_shape, classes, depth=16, dropout=1e-3, include_top=True):
    """
    feature_layer = "block5_pool"

    weight_layer = "predictions"
    """

    if depth is 16:
        base_model = applications.vgg16.VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling="avg")
    else:
        base_model = applications.vgg19.VGG19(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling="avg")
    x = base_model.output
    if include_top is True:
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def resnet50GAP(input_shape, classes, dropout=1e-3, include_top=True):
    """
    feature_layer = "activation_98"

    weight_layer = "predictions"
    """
    base_model = applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling="avg")
    x = base_model.output

    if include_top is True:
        x = Dropout(dropout, name='dropout')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def inceptionV3GAP(input_shape, include_top=True, classes=1000, dropout=1e-3,):
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

    if include_top is True:
        x = Dropout(dropout, name='dropout')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def inceptionResnetV2GAP(input_shape, classes, dropout=1e-3, include_top=True):
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

    if include_top is True:
        x = Dropout(dropout, name='dropout')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def mobilenetGAP(input_shape, classes, dropout=1e-3, include_top=True, finalAct="softmax"):
    """在 ImageNet 上预训练的 MobileNet 模型。

    Args:
        input_shape (:obj:`str`): 模型的输入尺寸。格式为 `(height, width, channel)`。
        classes (:obj:`turple` of :obj:`int`, 可选): 所有的图像都将以该尺寸读入内存。格式为 `(height, width, channel)`，默认为`(224, 336, 3)`。
        dropout (:obj:`bool`, 可选): 是否读取增强数据。
        include_top (:obj:`bool`, 可选): 是否读取增强数据。

    Note:
        要通过 `load_model` 载入 MobileNet 模型，你需要导入自定义对象 `relu6` 和 `DepthwiseConv2D` 并通过 `custom_objects` 传参。例如 ::

            model = load_model('mobilenet.h5', custom_objects={
                    'relu6': mobilenet.relu6,
                    'DepthwiseConv2D': mobilenet.DepthwiseConv2D})

    feature_layer = "conv_pw_13_relu"

    weight_layer = "predictions"


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

    if include_top is True:
        #    x = Reshape((1, 1, 1024), name='reshape_1')(x)
        x = Dropout(dropout, name='dropout')(x)
    #    x = Conv2D(classes, (1, 1),
    #               padding='same', name='conv_preds')(x)
    #    x = Activation('softmax', name='act_softmax')(x)
    #    x = Reshape((classes,), name='reshape')(x)
        x = Dense(classes, activation=finalAct, name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def denseNetGAP(input_shape, classes, depth, dropout=1e-3, include_top=True):
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

    if include_top is True:
        x = Dropout(dropout, name='dropout')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def NASNetGAP(input_shape, classes, MorL, dropout=1e-3, include_top=True):
    """
    feature_layer = "activation_376"

    feature_layer = "activation_520"

    weight_layer = "predictions"
    """

    if MorL is "mobile":
        base_conv = applications.nasnet.NASNetMobile(
            include_top=False,
            weights='imagenet',
            pooling="avg")
        base_model = applications.nasnet.NASNetMobile(
            input_shape=input_shape,
            include_top=False,
            weights=None,
            pooling="avg")
    else:
        base_conv = applications.nasnet.NASNetLarge(
            include_top=False,
            weights='imagenet',
            pooling="avg")
        base_model = applications.nasnet.NASNetLarge(
            input_shape=input_shape,
            include_top=False,
            weights=None,
            pooling="avg")
    for new_layer, layer in zip(base_model.layers[0:], base_conv.layers[0:]):
        new_layer.set_weights(layer.get_weights())

    x = base_model.output

    if include_top is True:
        x = Dropout(dropout, name='dropout')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def top(input_shape, classes, dropout=1e-3, finalAct="softmax"):
    """通用top层。

    如果模型去掉了顶层的全连接层，可以用 :meth:`top` 来衔接得到分类结果。

    Args:
        path (:obj:`str`): 数据所在目录地址，或 :obj:`h5` 文件地址。
        input_shape (:obj:`turple` of :obj:`int`, 可选): 所有的图像都将以该尺寸读入内存。格式为 `(height, width, channel)`，默认为`(224, 336, 3)`。
        augment (:obj:`bool`, 可选): 是否读取增强数据。

    """
    model = Sequential()

    model.add(Dropout(dropout, input_shape=input_shape, name='dropout1'))
#    model.add(Dense(4096, input_shape=input_shape, activation='relu', name='fc1'))
#    model.add(Dense(4096, activation='relu', name='fc2'))

    model.add(Dense(
        units=classes,
        activation=finalAct,
        name='predictions'))


    return model

def resnet50Triplet(input_shape, classes, dropout=0.5, finalAct="softmax"):
    """用于三元组损失
    """
    base_model = resnet50GAP(
    input_shape=input_shape,
    classes=classes,
    dropout=dropout)

    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')
    encoded_anchor = base_model(anchor_input)
    encoded_positive = base_model(positive_input)
    encoded_negative = base_model(negative_input)
    merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

    pred_model = Model(input=anchor_input, output=encoded_anchor)
    feature_model = Model(input=base_model.input, output=base_model.output)
    class_triplet_model = Model(
            input=[anchor_input, positive_input, negative_input],
            output=[base_model.output, merged_vector])
    return pred_model, feature_model, class_triplet_model



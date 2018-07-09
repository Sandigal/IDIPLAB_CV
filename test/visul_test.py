
import dataset_io as io
import visul

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from pickle import load
import imgaug as ia
from imgaug import augmenters as iaa

from keras.models import load_model, Model
from keras.applications.mobilenet import relu6
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils.generic_utils import CustomObjectScope

# %%

path = "../data/dataset 336x224"
shape = (336, 224)
dataset = io.Dataset(path, shape=shape)
class_to_index, sample_per_class = dataset.load_data()
del path,shape

#imgs_origin = np.array(dataset.imgs_origin)
#labels_origin = dataset.labels_origin
#names_origin = dataset.names_origin
#del path, shape

test_shape=0.2
_, _, imgs_test, labels_test = dataset.train_test_split(
    test_shape=test_shape)
del test_shape

total_splits=3
valid_split=0
imgs_train, labels_train, imgs_valid, labels_valid = dataset.cross_split(
    total_splits=total_splits, valid_split=valid_split)
del total_splits,valid_split

mean, std = load(open('mean-std.json', 'rb'))
imgs_white = (imgs_valid-mean)/std
del mean, std

#%%

visul.showGrid(imgs_train[0:25])

#%%

with CustomObjectScope({'relu6': relu6}):
    model = load_model('2018-06-22 model.h5')
model.summary()

# %%

index = 0
img_white = imgs_white[index]
img_show = imgs_valid[index]
label_show = labels_valid[index]
feature_layer = 'conv_pw_13_relu'
weight_layer = 'conv_preds'

cam, mix = visul.CAM(
    img_white=img_white,
    model=model,
    feature_layer=feature_layer,
    weight_layer=weight_layer,
    display=True,
    img_show=img_show,
    label_show=label_show,
    class_to_index=class_to_index,
    top2=True)
#xAB, yAB, wAB, hAB, xSC, ySC, xxSC, yySC = visul.cropMask(
#        cam=cam,
#        img_show=mix,
#        display=True)

#%%

plt.ioff()
process_bar = io.ShowProcess(len(imgs_valid))
for i in range(len(imgs_valid)):
    process_bar.show_process()

    index = i
    img_white = imgs_white[index]
    img_show = imgs_valid[index]
    label_show = labels_valid[index]
    feature_layer = 'conv_pw_13_relu'
    weight_layer = 'conv_preds'

    cam, mix = visul.CAM(
        img_white=img_white,
        model=model,
        feature_layer=feature_layer,
        weight_layer=weight_layer,
        display=True,
        img_show=img_show,
        label_show=label_show,
        class_to_index=class_to_index,
        top2=True)


# %%

imgs = []
for i in range(0, len(labels_origin), len(labels_origin)//9):
    print(i)
    index = i
    img_white = imgs_white[index]
    _, mix = visul.CAM(img_white=img_white, model=model, feature_layer='conv_pw_13_relu',
                       weight_layer='conv_preds')
    mix = Image.fromarray(mix)
    imgs.append(mix)

grid = visul.showGrid(imgs)
# grid.save("25.jpg")
plt.imshow(grid)


#%%

path="../训练结果/7.3 第一步 宽图像 better/2018-07-03 history.json"
visul.showHistory(path)

#%%

visul.showHistorys()

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:37:39 2018

@author: Sandiagal
"""

from dataset_io import reverse_dict

import numpy as np
import math
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
from pickle import load

from keras import backend as K
from keras.models import Model

# %%


def showHistory(path):

    history = load(open(path, 'rb'))

    plt.figure(1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Train loss: %.4f - Val loss: %.4f' %
              (history['loss'][-1], history['val_loss'][-1]))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.ylim((-.05, 5))
    plt.show()
    plt.savefig('train val loss.jpg')

    plt.figure(2)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Train acc: %.4f - Valid acc: %.4f' %
              (history['acc'][-1], history['val_acc'][-1]))
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc'], loc='upper right')
    plt.ylim((0, 1.05))
    plt.show()
    plt.savefig('train val acc.jpg')

    return plt


def showHistorys(path=""):

    accs = []
    val_accs = []
    for i in range(3):
        f = open(path+"20180704_validSplit."+str(i)+"_result.h5", "rb")
        contact = load(f)
        accs.append(contact["history"]["acc"])
        val_accs.append(contact["history"]["val_acc"])

    plt.figure()
    len_acc = np.max([len(acc) for acc in accs])
    for i in range(len(accs)):
        accs[i] = accs[i]+(len_acc-len(accs[i]))*[accs[i][-1]]
        val_accs[i] = val_accs[i]+(len_acc-len(val_accs[i]))*[val_accs[i][-1]]
        plt.plot(val_accs[i], lw=1, alpha=0.3, label="Accuracy for %s fold (%.2f%%)" % (i, val_accs[i][-1]*100))

    accs_mean = np.mean(accs, axis=0)
    accs_std = np.std(accs, axis=0)
    val_accs_mean = np.mean(val_accs, axis=0)
    val_accs_std = np.std(val_accs, axis=0)

    plt.fill_between(range(len_acc), accs_mean - accs_std,
                     accs_mean + accs_std, alpha=0.2,
                     color="r")
    plt.fill_between(range(len_acc), val_accs_mean - val_accs_std,
                     val_accs_mean + val_accs_std, alpha=0.2, color="g")
    plt.plot(accs_mean, "o-", alpha=0.8, color="r",
             label="Training score")
    plt.plot(val_accs_mean, "o-", alpha=0.8, color="g",
             label="Cross-validation score")

    plt.grid()
    plt.title("Learning Curves \nTrain: %.2f%% - Valid: %.2f%%" %
              (accs_mean[-1]*100, val_accs_mean[-1]*100))
    plt.ylim(0, 1.05)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="best")
    plt.savefig("train val acc.jpg")

    return plt


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i/dist
        x = int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y = int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def drawpoly(img, pts, color, thickness=1, style='dotted',):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)


def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)

# %%


def show_grid(imgs, title=None, suptitles=None):
    imgs_num = len(imgs)

    if type(imgs[0]) == np.ndarray:
        #        imgsPIL=[]
        #        for i in range(len(imgs)):
        #            imgsPIL.append(Image.fromarray(imgs[i]))
        imgs = [Image.fromarray(img) for img in imgs]
#    imgs=imgsPIL

#    width, height = 0, 0
#    for i in range(len(imgs)):
#        w, h = imgs[i].size
#        if width < w:
#            width = w
#        if height < h:
#            height = h

    rows = math.ceil(np.sqrt(imgs_num))
    cols = int(round(np.sqrt(imgs_num)))
#    space = 2*int(np.sqrt(width*height)/40)

#    newSize = ((width+space) * cols, (height+space) * cols)
#    emphImf = Image.new('RGB', (width+space, width+space), (225, 225, 0))
#    grid = Image.new('RGB', newSize)

#    plt.figure()
    f, axarr = plt.subplots(cols, rows,gridspec_kw={"wspace": 0, "hspace": .5})
#                            figsize=(15,15),
    if title:
        plt.suptitle(title)
    for idx, ax in enumerate(f.axes):
        if idx < imgs_num:
            ax.imshow(imgs[idx])
            if suptitles:
                ax.set_title(suptitles[idx])
        ax.axis("off")

#    for y in range(rows):
#        for x in range(cols):
#            curImage = imgs[cols*y+x]
#            grid.paste(curImage, (x * width + int((x+0.5) * space),
#                                  y * height + int((y+0.5) * space)))
#
#    grid.resize((int(newSize[0]*1080/newSize[1]), 1080), Image.ANTIALIAS)

#    plt.imshow(grid)
#    plt.axis('off')
    return plt

def overall(imgs, number_to_show=20):
    # Plot images of the digits
    img_show = []
    for i in range(0, len(imgs), len(imgs)//number_to_show):
        img_show.append(imgs[i])
    grid = showGrid(img_show)
    plt.figure(figsize=(10, 9))
    plt.imshow(grid)
    plt.title('A selection from the dataset')
    plt.xticks([])
    plt.yticks([])


def CAM(img_white, model, feature_layer, weight_layer, idx_predic=None, display=False, img_show=None, label_show=None, class_to_index=None, top2=False):
    width, height, _ = img_white.shape

    getOutput = K.function([model.input], [model.get_layer(
        feature_layer).output, model.output])
    [avtiveMap, scores_predict] = getOutput([np.expand_dims(img_white, axis=0)])
    if idx_predic == None:
        idx_predic = np.argmax(scores_predict)

    avtiveMap = avtiveMap[0]

    weightLayer = model.get_layer(weight_layer)
    weightsClasses = weightLayer.get_weights()[0]

    weightsClass = weightsClasses[0, 0, :, idx_predic]

    cam = np.matmul(avtiveMap, weightsClass)

    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = 1-cam
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam > 0.95)] = 0

    if display:
        mix = cv2.addWeighted(src1=img_show, src2=heatmap,
                              alpha=0.8, beta=0.4, gamma=0)

        index_to_class = reverse_dict(class_to_index)
        predic_class = index_to_class[idx_predic]
        predict_score = np.max(scores_predict)

        if top2 is False:

            plt.figure(figsize=(11, 8))
            plt.subplot(121)
            plt.axis('off')
            plt.title("Actual: %s" % (label_show))
            plt.imshow(img_show)
            plt.subplot(122)
            plt.axis('off')
            plt.title("Predict: %s %.2f%%" %
                      (predic_class, predict_score * 100))
            plt.imshow(mix)

        else:

            plt.figure()
            plt.subplot(221)
            plt.axis('off')
            plt.title("Original image: %s" % (label_show))
            plt.imshow(img_show)

            plt.subplot(222)
            plt.axis('off')
            plt.title("Top 1: %s %.2f%%" % (predic_class, predict_score * 100))
            plt.imshow(mix)

            idx_predic3 = class_to_index[label_show]
            predict_score3 = scores_predict[0, idx_predic3]
            weightsClass3 = weightsClasses[0, 0, :, idx_predic3]
            cam3 = np.matmul(avtiveMap, weightsClass3)
            cam3 = (cam3 - cam3.min()) / (cam3.max() - cam3.min())
            cam3 = 1-cam3
            cam3 = cv2.resize(cam3, (height, width))
            heatmap3 = cv2.applyColorMap(np.uint8(255*cam3), cv2.COLORMAP_JET)
            heatmap3[np.where(cam3 > 0.95)] = 0
            mix3 = cv2.addWeighted(src1=img_show, src2=heatmap3,
                                   alpha=0.8, beta=0.4, gamma=0)

            plt.subplot(223)
            plt.axis('off')
            plt.title("For ground truth: %s %.2f%%" %
                      (label_show, predict_score3 * 100))
            plt.imshow(mix3)

            idx_predic4 = np.argsort(scores_predict[0, :])[-2]
            predic_class4 = index_to_class[idx_predic4]
            predict_score4 = scores_predict[0, idx_predic4]
            weightsClass4 = weightsClasses[0, 0, :, idx_predic4]
            cam4 = np.matmul(avtiveMap, weightsClass4)
            cam4 = (cam4 - cam4.min()) / (cam4.max() - cam4.min())
            cam4 = 1-cam4
            cam4 = cv2.resize(cam4, (height, width))
            heatmap4 = cv2.applyColorMap(np.uint8(255*cam4), cv2.COLORMAP_JET)
            heatmap4[np.where(cam4 > 0.95)] = 0
            mix4 = cv2.addWeighted(src1=img_show, src2=heatmap4,
                                   alpha=0.8, beta=0.4, gamma=0)

            plt.subplot(224)
            plt.axis('off')
            plt.title("Top 2: %s %.2f%%" %
                      (predic_class4, predict_score4 * 100))
            plt.imshow(mix4)

            plt.savefig("True.%s(%.1f%%) Top1.%s(%.1f%%) Top2.%s(%.1f%%).jpg" %
                        (
                            #                                label_show,
                            label_show, predict_score3 * 100,
                            predic_class, predict_score * 100,
                            predic_class4, predict_score4 * 100))

        return cam, mix
    return cam


def CAMs(imgs_white, model, feature_layer, weight_layer, idxs_predic=None):
    sample, width, height, _ = imgs_white.shape
    if idxs_predic == None:
        idxs_predic = [None]*sample

    getFeatureMaps = Model(inputs=model.input, outputs=model.get_layer(
        feature_layer).output)
    feature_maps = getFeatureMaps.predict(imgs_white, batch_size=32, verbose=1)

    getScoresPredict = K.function([model.get_layer(index=model.layers.index(
        model.get_layer(feature_layer))+1).input], [model.output])
    [scores_predict] = getScoresPredict([feature_maps])

    weightLayer = model.get_layer(weight_layer)
    weightsClasses = weightLayer.get_weights()[0]

    cams = []
    for i in range(sample):

        if idxs_predic[i] == None:
            idxs_predic[i] = np.argmax(scores_predict[i], axis=1)
        weightsClass = weightsClasses[0, 0, i, idxs_predic[i]]

        cam = feature_maps[i]@ weightsClass[i]
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = 1-cam
        cam = cv2.resize(cam, (height, width))
        cams.append(cam)
    return cams


def cropMask(cam, img_show, display=False):
    img_show = np.copy(img_show)
    n, h,  = cam.shape

    can = 255*(1-cam)
    can = can.astype('uint8')
    _, thresh = cv2.threshold(can, 0.7*255, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_Index = 0
    for i in range(len(contours)):
        max_Index = i if contours[i].shape[0] > contours[max_Index].shape[0] else max_Index
    cnt = contours[max_Index]
    xHRO, yHRO, wHRO, hHRO = cv2.boundingRect(cnt)
#    plt.figure()
#    plt.imshow(thresh)

    _, thresh = cv2.threshold(can, 0.4*255, 255, cv2.THRESH_BINARY)
    _, contoursAB, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contoursAB:
        if np.min(contour[:, 0, 0]) <= np.min(cnt[:, 0, 0]) and np.max(contour[:, 0, 0]) >= np.max(cnt[:, 0, 0]) and np.min(contour[:, 0, 1]) <= np.min(cnt[:, 0, 1]) and np.max(contour[:, 0, 1]) >= np.max(cnt[:, 0, 1]):
            cnt = contour
    xAB, yAB, wAB, hAB = cv2.boundingRect(cnt)
#    plt.figure()
#    plt.imshow(thresh)
#    return contours, contours1

    #xx = 5
    #yy = 70
    #cv2.rectangle(target, (xx, yy), (xx+224, yy+224), (192,192,192), 5)
    #text = "Random crop"
    # cv2.putText(target, text, (xx, yy), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #            fontScale=0.5, color=(192,192,192), thickness=10, lineType=cv2.LINE_AA)
    # cv2.putText(target, text, (xx, yy), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #            fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    xSC = max(xHRO+wHRO-wAB, 0)
    ySC = max(yHRO+hHRO-hAB, 0)
    ww = min(xHRO+wAB, h)
    hh = min(yHRO+hAB, n)

    if display:
        cv2.rectangle(img_show, (xSC, ySC), (xSC+wAB, ySC+hAB), (0, 255, 0), 5)
        drawrect(img_show, (xSC, ySC), (ww, hh), (0, 255, 0), 5, 'dotted')
        text = "Supervised crop"
        cv2.putText(img_show, text, (xSC, ySC), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 255, 0), thickness=10, lineType=cv2.LINE_AA)
        cv2.putText(img_show, text, (xSC, ySC), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 0, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.rectangle(img_show, (xAB, yAB), (xAB+wAB, yAB+hAB), (0, 0, 255), 5)
        text = "Anchor Box (%dx%d)" % (wAB, hAB)
        cv2.putText(img_show, text, (xAB, yAB), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 0, 255), thickness=10, lineType=cv2.LINE_AA)
        cv2.putText(img_show, text, (xAB, yAB), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.rectangle(img_show, (xHRO, yHRO),
                      (xHRO+wHRO, yHRO+hHRO), (220, 20, 60), 5)
        text = "Highest respond area (%dx%d)" % (wHRO, hHRO)
        cv2.putText(img_show, text, (xHRO, yHRO), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(220, 20, 60), thickness=10, lineType=cv2.LINE_AA)
        cv2.putText(img_show, text, (xHRO, yHRO), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(35, 235, 195), thickness=1, lineType=cv2.LINE_AA)

        plt.figure(figsize=(10, 8))
        plt.imshow(img_show)
        plt.axis('off')
#        plt.savefig('crop.jpg')

    return xAB, yAB, wAB, hAB, xSC, ySC, ww-wAB, hh-hAB

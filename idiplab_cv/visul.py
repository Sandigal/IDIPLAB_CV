"""
该模块:`visul`包含数据可视化的类和函数。
"""

# Author: Sandiagal <sandiagal2525@gmail.com>,
# License: GPL-3.0

import math
from pickle import load
import os

import cv2
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

from idiplab_cv.dataset_io import reverse_dict
from idiplab_cv.dataset_io import to_categorical

if not os.path.exists("images"):
    os.mkdir("images")

# %%


def _drawline(img, pt1, pt2, color, thickness=1, style="dotted", gap=20):
    dist = ((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i/dist
        x = int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y = int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x, y)
        pts.append(p)

    if style == "dotted":
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


def _drawpoly(img, pts, color, thickness=1, style="dotted",):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        _drawline(img, s, e, color, thickness, style)


def _drawrect(img, pt1, pt2, color, thickness=1, style="dotted"):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    _drawpoly(img, pts, color, thickness, style)

# %%


def show_grid(imgs, title=None, suptitles=None, rows=None, cols=None, colors=None):

    plt.style.use("classic")

    imgs_num = len(imgs)

    imgs = np.array(imgs)
    if type(imgs[0]) == np.ndarray:
        imgs = [Image.fromarray(img.astype('uint8')) for img in imgs]

    if not rows:
        rows = math.ceil(np.sqrt(imgs_num))
    if not cols:
        cols = int(round(np.sqrt(imgs_num)))

    f, axarr = plt.subplots(cols, rows,
                            figsize=(15, 20),
                            gridspec_kw={"wspace": 0.03, "hspace": 0.05})
#    if title:
#        plt.suptitle(title, fontsize=20)

    for idx, ax in enumerate(f.axes):
        if idx < imgs_num:
            ax.imshow(imgs[idx])
            if suptitles:
                if colors:
                    ax.set_title(suptitles[idx], ha='center',
                                 fontsize=14, color=colors[idx])
                else:
                    ax.set_title(suptitles[idx], fontsize=10)
        ax.axis("off")
    if not title:
        title = "grid"
    plt.savefig("images/"+title+".png", bbox_inches='tight', dpi=200)

    return plt


def overall(imgs, number_to_show=20):
    # Plot images of the digits
    img_show = []
    for i in range(0, len(imgs), len(imgs)//number_to_show):
        img_show.append(imgs[i])
    plt = show_grid(img_show, "A selection from the dataset")
    return plt


def CAM(img_white=None, model=None, feature_layer=None, weight_layer=None, feature_map=None, weights=None, scores_predict=None, idx_predic=None, display=False, img_show=None, label_show=None, class_to_index=None, extend=False):

    width, height, _ = img_show.shape
    omit = 1.75

    if feature_map is None or weights is None or scores_predict is None:

        getOutput = K.function([model.input], [model.get_layer(
            feature_layer).output, model.output])
        [feature_map, scores_predict] = getOutput(
            [np.expand_dims(img_white, axis=0)])

        weightLayer = model.get_layer(weight_layer)
        weights = weightLayer.get_weights()[0]

    if idx_predic == None:
        idx_predic = np.argmax(scores_predict)
    weight = weights[:, idx_predic]
    feature_map = feature_map[0]
    cam = np.matmul(feature_map, weight)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    if cam[0, 0]+cam[0, -1]+cam[-1, 0]+cam[-1, -1] < 3:
        cam = 1-cam
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam > omit)] = 0

    mix = cv2.addWeighted(src1=img_show.astype("uint8"), src2=heatmap,
                          alpha=0.8, beta=0.4, gamma=0)
    if display:

        Image.fromarray(img_show).save('images/img_origin.png')
        Image.fromarray(mix).save('images/CAM.png')

        index_to_class = reverse_dict(class_to_index)
        predic_class = index_to_class[idx_predic]
        predict_score = np.max(scores_predict)

        if extend is False:

            plt.figure(figsize=(11, 8))
            plt.subplot(121)
            plt.axis("off")
            plt.title("Actual: %s" % (label_show))
            plt.imshow(img_show)
            plt.subplot(122)
            plt.axis("off")
            plt.title("Predict: %s %.2f%%" %
                      (predic_class, predict_score * 100))
            plt.imshow(mix)

        else:

            plt.figure()
            plt.subplot(221)
            plt.axis("off")
            plt.title("Original image: %s" % (index_to_class[label_show]))
            plt.imshow(img_show)

            plt.subplot(222)
            plt.axis("off")
            plt.title("Top 1: %s %.2f%%" % (predic_class, predict_score * 100))
            plt.imshow(mix)

            idx_predic3 = label_show
            predict_score3 = scores_predict[0, idx_predic3]
            weight3 = weights[:, idx_predic3]
            cam3 = np.matmul(feature_map, weight3)
            cam3 = (cam3 - cam3.min()) / (cam3.max() - cam3.min())
            if cam3[0, 0]+cam3[0, -1]+cam3[-1, 0]+cam3[-1, -1] < 2:
                cam3 = 1-cam3
            cam3 = cv2.resize(cam3, (height, width))
            heatmap3 = cv2.applyColorMap(np.uint8(255*cam3), cv2.COLORMAP_JET)
            heatmap3[np.where(cam3 > omit)] = 0
            mix3 = cv2.addWeighted(src1=img_show, src2=heatmap3,
                                   alpha=0.8, beta=0.4, gamma=0)

            plt.subplot(223)
            plt.axis("off")
            plt.title("For ground truth: %s %.2f%%" %
                      (index_to_class[label_show], predict_score3 * 100))
            plt.imshow(mix3)

            idx_predic4 = np.argsort(scores_predict[0, :])[-2]
            predic_class4 = index_to_class[idx_predic4]
            predict_score4 = scores_predict[0, idx_predic4]
            weight4 = weights[:, idx_predic4]
            cam4 = np.matmul(feature_map, weight4)
            cam4 = (cam4 - cam4.min()) / (cam4.max() - cam4.min())
            if cam4[0, 0]+cam4[0, -1]+cam4[-1, 0]+cam4[-1, -1] < 2:
                cam4 = 1-cam4
            cam4 = cv2.resize(cam4, (height, width))
            heatmap4 = cv2.applyColorMap(np.uint8(255*cam4), cv2.COLORMAP_JET)
            heatmap4[np.where(cam4 > omit)] = 0
            mix4 = cv2.addWeighted(src1=img_show, src2=heatmap4,
                                   alpha=0.8, beta=0.4, gamma=0)

            plt.subplot(224)
            plt.axis("off")
            plt.title("Top 2: %s %.2f%%" %
                      (predic_class4, predict_score4 * 100))
            plt.imshow(mix4)

            plt.savefig("images/"+"True.%s(%.1f%%) Top1.%s(%.1f%%) Top2.%s(%.1f%%).jpg" %
                        (
                            #                                label_show,
                            label_show, predict_score3 * 100,
                            predic_class, predict_score * 100,
                            predic_class4, predict_score4 * 100),
                        bbox_inches='tight',
                        dpi=300
                        )

    return cam, mix


def CAMs(imgs_white, model, feature_layer, weight_layer, idxs_predic=None, display=False, img_show=None):
    '''
    一次性计算所有图像的CAM
    '''
    if not isinstance(imgs_white, dict):
        sample, width, height, _ = imgs_white.shape

        getFeatureMaps = Model(inputs=model.input, outputs=model.get_layer(
            feature_layer).output)
        feature_maps = getFeatureMaps.predict(
            imgs_white, batch_size=32, verbose=1)

        getScoresPredict = K.function([model.get_layer(index=model.layers.index(
            model.get_layer(feature_layer))+1).input], [model.output])
        [scores_predict] = getScoresPredict([feature_maps])

        weightLayer = model.get_layer(weight_layer)
        weights = weightLayer.get_weights()[0]

    else:
        sample = len(imgs_white["feature_maps"])
        _, width, height, _ = model.input.shape.as_list()
        feature_maps = imgs_white["feature_maps"]
        weights = imgs_white["weights"]

        getScoresPredict = K.function([model.get_layer(index=model.layers.index(
            model.get_layer(feature_layer))+1).input], [model.output])
        [scores_predict] = getScoresPredict([feature_maps])

    if idxs_predic == None:
        idxs_predic = [None]*sample

    cams = []
    for i in range(sample):

        if idxs_predic[i] == None:
            idxs_predic[i] = np.argmax(scores_predict[i])
        cam = feature_maps[i]@ weights[:, idxs_predic[i]]
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        if cam[0, 0]+cam[0, -1]+cam[-1, 0]+cam[-1, -1] < 2:
            cam = 1-cam
        cam = cv2.resize(cam, (height, width))
        cams.append(cam)

        if display:
            if i == 0:
                heatmap = cv2.applyColorMap(
                    np.uint8(255*cam), cv2.COLORMAP_JET)
                heatmap[np.where(cam > 0.8)] = 0
                mix = cv2.addWeighted(src1=img_show, src2=heatmap,
                                      alpha=0.8, beta=0.4, gamma=0)
                plt.figure(figsize=(11, 8))
                plt.imshow(mix)

    return cams


def cropMask(cam, img_show, display=False):
    img_show = np.copy(img_show)
    cam = np.copy(cam)
    n, h,  = cam.shape

    cam = 255*(1-cam)
    cam = cam.astype("uint8")
    _, thresh = cv2.threshold(cam, 0.7*255, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_Index = 0
    for i in range(len(contours)):
        max_Index = i if contours[i].shape[0] > contours[max_Index].shape[0] else max_Index
    cnt = contours[max_Index]
    xHRO, yHRO, wHRO, hHRO = cv2.boundingRect(cnt)

    _, thresh = cv2.threshold(cam, 0.4*255, 255, cv2.THRESH_BINARY)
    contoursAB, hierarchy = cv2.findContours(
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
        _drawrect(img_show, (xSC, ySC), (ww, hh), (0, 255, 0), 5, "dotted")
        text = "Supervised Crop Box (%dx%d)" % (wAB, hAB)
        cv2.putText(img_show, text, (xSC, ySC), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=(0, 255, 0), thickness=10, lineType=cv2.LINE_AA)
        cv2.putText(img_show, text, (xSC, ySC), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.rectangle(img_show, (xAB, yAB), (xAB+wAB, yAB+hAB), (0, 0, 255), 5)
        text = "Anchor Box (%dx%d)" % (wAB, hAB)
        cv2.putText(img_show, text, (xAB, yAB), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=(0, 0, 255), thickness=10, lineType=cv2.LINE_AA)
        cv2.putText(img_show, text, (xAB, yAB), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.rectangle(img_show, (xHRO, yHRO),
                      (xHRO+wHRO, yHRO+hHRO), (220, 20, 60), 5)
        text = "Highest Respond Box (%dx%d)" % (wHRO, hHRO)
        cv2.putText(img_show, text, (xHRO, yHRO), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=(220, 20, 60), thickness=10, lineType=cv2.LINE_AA)
        cv2.putText(img_show, text, (xHRO, yHRO), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        plt.figure(figsize=(10, 8))
        plt.imshow(img_show)
        plt.axis("off")
        Image.fromarray(img_show).save('images/CAM_ABSC.png')

    return xAB, yAB, wAB, hAB, xSC, ySC, ww-wAB, hh-hAB

def worst_samples(imgs_valid, labels_valid, score_predict, class_to_index, top=10, names_valid=None):
    '''
    栅格展示分类效果最差的样本
    '''
    index_to_class = reverse_dict(class_to_index)
    labels_valid_onehot = to_categorical(labels_valid, len(class_to_index))
    score_valid = np.max(np.array(score_predict)*labels_valid_onehot, axis=1)
    labels_predict = np.argmax(score_predict, axis=1)
    score_predict_max = np.max(score_predict, axis=1)
    worst_index = np.argsort(-score_predict_max+score_valid)[:top]
    asd = score_predict_max-score_valid

    imgs = []
    suptitles = []
    for _, index in enumerate(worst_index):
        imgs.append(imgs_valid[index])
        suptitle = 'Predict: %s (%5.2f%%)\n True  : %s (%5.2f%%)' % (
            index_to_class[labels_predict[index]],
            score_predict_max[index]*100,
            index_to_class[labels_valid[index]],
            score_valid[index]*100)
        suptitles.append(suptitle)
        if names_valid is not None:
            name = 'Predict %s True %s' % (
                index_to_class[labels_predict[index]],
                index_to_class[labels_valid[index]])
            print('%5.2f' % (asd[index]*100))
            print(name+'\n'+names_valid[index])
            print()
    plt = show_grid(imgs, 'Worst prediction samples', suptitles=suptitles)
    return plt

def best_worst_samples(imgs_valid, labels_valid, feature_maps, weights, scores_predict, class_to_index):
    '''
    模型分级结果和对应的CAM的示例。
    同一行的示例具有相同的真实等级。
    每一列中，前两副图表示最好的预测结果，而后两副图表示最差的结果。
    在每一个示例中，我们首先展示了原始图像的真实等级和治疗对策，然后标出了预测等级和治疗建议，并标注了对应的预测概率和对应等级的CAM。
    CAM中越红的位置表示其越有可能是病灶区域。
    '''
    index_to_class = reverse_dict(class_to_index)
    labels_valid_onehot = to_categorical(labels_valid, len(class_to_index))
    score_valid = np.max(np.array(scores_predict)*labels_valid_onehot, axis=1)
    labels_predict = np.argmax(scores_predict, axis=1)
    score_predict_max = np.max(scores_predict, axis=1)

    imgs = []
    suptitles = []
    colors = []
    for i in range(0, len(class_to_index)):
        #    i=2
        cur_class = labels_valid == i
        cur_class = np.array([i for i, _ in enumerate(labels_valid)])[
            cur_class]
        cur_score_predict = score_predict_max[cur_class]
        cur_score_valid = score_valid[cur_class]
        worst_index = np.argsort(-cur_score_predict+cur_score_valid)[:1]
        best_index = np.argsort(-cur_score_predict+cur_score_valid)[-1:]

        for _, index in enumerate(cur_class[best_index]):
            imgs.append(imgs_valid[index])
            suptitle = "True = %s\n%s" % (
                index_to_class[labels_valid[index]],
                "Follow-up" if (labels_valid[index]) < 2 else "Surgery")
            suptitles.append(suptitle)
            colors.append("blue")

            cam, mix = CAM(
                feature_map=np.expand_dims(feature_maps[index], axis=0),
                weights=weights,
                scores_predict=scores_predict[index],
                display=False,
                img_show=imgs_valid[index],
            )
            imgs.append(mix)
            suptitle = "Predict = %s (%4.1f%%)\n%s (%4.1f%%)" % (
                index_to_class[labels_predict[index]],
                score_predict_max[index]*100,
                "Follow-up" if (labels_predict[index]) < 2 else "Surgery",
                (sum(scores_predict[index, :2]) if (labels_predict[index]) < 2 else sum(scores_predict[index, 2:]))*100)
            suptitles.append(suptitle)
            colors.append("blue")

        for _, index in enumerate(cur_class[worst_index]):
            imgs.append(imgs_valid[index])
            suptitle = "True = %s\n%s" % (
                index_to_class[labels_valid[index]],
                "Follow-up" if (labels_valid[index]) < 2 else "Surgery")
            suptitles.append(suptitle)
            colors.append("red")

            cam, mix = CAM(
                feature_map=np.expand_dims(feature_maps[index], axis=0),
                weights=weights,
                scores_predict=scores_predict[index],
                display=False,
                img_show=imgs_valid[index],
            )
            imgs.append(mix)
            suptitle = "Predict = %s (%4.1f%%)\n%s (%4.1f%%)" % (
                index_to_class[labels_predict[index]],
                score_predict_max[index]*100,
                "Follow-up" if (labels_predict[index]) < 2 else "Surgery",
                (sum(scores_predict[index, :2]) if (labels_predict[index]) < 2 else sum(scores_predict[index, 2:]))*100)
            suptitles.append(suptitle)
            colors.append("red")

    plt = show_grid(imgs, suptitles=suptitles, rows=4, cols=6, colors=colors, title="best_worst_samples")
    return plt

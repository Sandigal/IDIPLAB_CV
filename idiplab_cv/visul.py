# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:37:39 2018

@author: Sandiagal
"""

import numpy as np
import cv2
from keras import backend as K
import matplotlib.pyplot as plt
import PIL.Image as Image

# %%


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


def showGrid(imgs, emphs=[]):
    imgs_num = len(imgs)

    width, height = 0, 0
    for i in range(len(imgs)):
        if type(imgs[i]) == np.ndarray:
            imgs[i] = Image.fromarray(imgs[i])
        w, h = imgs[i].size
        if width < w:
            width = w
        if height < h:
            height = h

    cols = int(np.sqrt(imgs_num))
    rows = int(0.5+np.sqrt(imgs_num))
    space = 2*int(np.sqrt(width*height)/40)

    newSize = ((width+space) * cols, (height+space) * cols)
    emphImf = Image.new('RGB', (width+space, width+space), (225, 225, 0))
    grid = Image.new('RGB', newSize)

    for i in emphs:
        x = i % cols
        y = i//rows
        grid.paste(emphImf, (x * (width + space), y * (height + space)))

    for y in range(rows):
        for x in range(cols):
            curImage = imgs[cols*y+x]
            grid.paste(curImage, (x * width + int((x+0.5) * space),
                                  y * height + int((y+0.5) * space)))

    grid.resize((int(newSize[0]*1080/newSize[1]), 1080), Image.ANTIALIAS)

    return grid


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


def CAM(img_white, model, active_layer, weight_layer, predicIdx=None, display=False, img_show=None):

    width, height, _ = img_white.shape

    getOutput = K.function([model.input], [model.get_layer(
        active_layer).output, model.output])
    [avtiveMap, predictions] = getOutput([np.expand_dims(img_white, axis=0)])
    if predicIdx == None:
        predicIdx = np.argmax(predictions)

    avtiveMap = avtiveMap[0]

    weightLayer = model.get_layer(weight_layer)
    weightsClasses = weightLayer.get_weights()[0]

    weightsClass = weightsClasses[0, 0, :, predicIdx]

    cam = np.matmul(avtiveMap, weightsClass)

    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = 1-cam
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam > 0.95)] = 0

    if display:
        mix = cv2.addWeighted(src1=img_show, src2=heatmap,
                              alpha=0.8, beta=0.4, gamma=0)

#        text = 'cat %.2f%%' % (100 - predict * 100)
#        cv2.putText(out, text, (100, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
#                color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)

        plt.figure()
        plt.subplot(121)
        plt.imshow(img_show)
        plt.subplot(122)
        plt.imshow(mix)
        plt.show()
        return cam, mix
    return cam


def cropMask(cam, target, display=False):
    target = np.copy(target)
    n, h,  = cam.shape

    can = 255*(1-cam)
    can = can.astype('uint8')
    _, thresh = cv2.threshold(can, 0.7*255, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    xHRO, yHRO, wHRO, hHRO = cv2.boundingRect(cnt)

    _, thresh = cv2.threshold(can, 0.4*255, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    xAB, yAB, wAB, hAB = cv2.boundingRect(cnt)

    #xx = 5
    #yy = 70
    #cv2.rectangle(target, (xx, yy), (xx+224, yy+224), (192,192,192), 5)
    #test = "Random crop"
    # cv2.putText(target, test, (xx, yy), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #            fontScale=0.5, color=(192,192,192), thickness=10, lineType=cv2.LINE_AA)
    # cv2.putText(target, test, (xx, yy), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #            fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    xSC = max(xHRO+wHRO-wAB, 0)
    ySC = max(yHRO+hHRO-hAB, 0)
    ww = min(xHRO+wAB, h)
    hh = min(yHRO+hAB, n)
    if display:
        cv2.rectangle(target, (xSC, ySC), (xSC+wAB, ySC+hAB), (0, 255, 0), 5)
        drawrect(target, (xSC, ySC), (ww, hh), (0, 255, 0), 5, 'dotted')
        test = "Supervised crop"
        cv2.putText(target, test, (xSC, ySC), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 255, 0), thickness=10, lineType=cv2.LINE_AA)
        cv2.putText(target, test, (xSC, ySC), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.rectangle(target, (xHRO, yHRO),
                      (xHRO+wHRO, yHRO+hHRO), (220, 20, 60), 5)
        test = "Highest respond area (%dx%d)" % (wHRO, hHRO)
        cv2.putText(target, test, (xHRO, yHRO), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(220, 20, 60), thickness=10, lineType=cv2.LINE_AA)
        cv2.putText(target, test, (xHRO, yHRO), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.rectangle(target, (xAB, yAB), (xAB+wAB, yAB+hAB), (0, 0, 255), 5)
        test = "Anchor Box (%dx%d)" % (wAB, hAB)
        cv2.putText(target, test, (xAB, yAB), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 0, 255), thickness=10, lineType=cv2.LINE_AA)
        cv2.putText(target, test, (xAB, yAB), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        plt.figure(figsize=(9, 6))
        plt.imshow(target)
        plt.axis('off')
#        plt.savefig('crop.jpg')

    return xAB, yAB, wAB, wAB, xSC, ySC, ww-wAB, hh-wAB

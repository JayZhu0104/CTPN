import sys
import os
import codecs
import cv2

import Net
from torch.utils.data import Dataset

def read_gt_file(path, have_BOM=False):
    """
    读取gt文件，返回gt的边框信息（每8个点表示一个box）
    """
    result = []
    if have_BOM:
        fp = codecs.open(path, 'r', 'utf-8-sig')
    else:
        fp = open(path, 'r')
    for line in fp.readlines():
        pt = line.split(',')
        # 8个数字表示一个box
        if have_BOM:
            box = [int(round(float(pt[i]))) for i in range(8)]
        else:
            box = [int(round(float(pt[i]))) for i in range(8)]
        result.append(box)
    fp.close()
    return result

def scale_img(img, gt, shortest_side=600):
    """
    对图片的处理：
    1.若图片的最短边小于600，需要将其放大
    2.图片放大后，还需要将其gt放大
    """
    height = img.shape[0]
    width = img.shape[1]
    scale = float(shortest_side) / float(min(height, width)) # 如果图片的最短边小于600，需要将其放大，这是放大比例
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    # 如果上面放大后，最短边还是小于600，则直接把最短边变为600
    if img.shape[0] < img.shape[1] and img.shape[0] != 600:
        img = cv2.resize(img, (600, img.shape[1]))
    elif img.shape[0] > img.shape[1] and img.shape[1] != 600:
        img = cv2.resize(img, (img.shape[0], 600))
    elif img.shape[0] != 600:
        img = cv2.resize(img, (600, 600))
    # 这是实际上的缩放比
    h_scale = float(img.shape[0]) / float(height)
    w_scale = float(img.shape[1]) / float(width)
    # 对gt也要进行一个缩放处理
    scale_gt = []
    for box in gt:
        scale_box = []
        for i in range(len(box)):
            if i % 2 == 0:
                scale_box.append(int(int(box[i]) * w_scale))
            else:
                scale_box.append(int(int(box[i]) * h_scale))
        scale_gt.append(scale_box)
    return img, scale_gt

def scale_img_only(img, shortest_side=600):
    """
    与上一个函数的区别是，该函数只针对图像大小进行处理
    而不用考虑gt的处理（测试集中）
    """
    height = img.shape[0]
    width = img.shape[1]
    scale = float(shortest_side)/float(min(height, width))
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    if img.shape[0] < img.shape[1] and img.shape[0] != 600:
        img = cv2.resize(img, (600, img.shape[1]))
    elif img.shape[0] > img.shape[1] and img.shape[1] != 600:
        img = cv2.resize(img, (img.shape[0], 600))
    elif img.shape[0] != 600:
        img = cv2.resize(img, (600, 600))

    return img
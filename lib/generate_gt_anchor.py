import sys
import math
import copy
import cv2
import time
import os

import draw_image


def sortCoords(box):
    """
    将可能随机排列的坐标点，统一重新排列成左下、左上、右上、右下排列
    """
    coords = [[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]]
    coords_x = [box[0], box[2], box[4], box[6]]
    coords_x.sort()
    coords_left = []
    coords_right = []
    for i in range(4):
        if coords[i][0] == coords_x[0] or coords[i][0] == coords_x[1]:
            coords_left.append(coords[i])
        else:
            coords_right.append(coords[i])
    new_box = []
    if coords_left[0][1] < coords_left[1][1]:
        new_box += coords_left[0]
        new_box += coords_left[1]
    else:
        new_box += coords_left[1]
        new_box += coords_left[0]
    if coords_right[0][1] > coords_right[1][1]:
        new_box += coords_right[0]
        new_box += coords_right[1]
    else:
        new_box += coords_right[1]
        new_box += coords_right[0]

    return new_box


def calcLine(box):
    """
    计算BBOX四条边所在的直线
    """
    x1 = box[0]
    x2 = box[2]
    x3 = box[4]
    x4 = box[6]
    y1 = box[1]
    y2 = box[3]
    y3 = box[5]
    y4 = box[7]

    # l12,左边
    a12 = y2 - y1
    b12 = x1 - x2
    c12 = x2 * y1 - x1 * y2
    line12 = [a12,b12,c12]

    # l34，右边
    a34 = y4 - y3
    b34 = x3 - x4
    c34 = x4 * y3 - x3 * y4
    line34 = [a34,b34,c34]

    # l14，左上角到右下角
    a14 = y4 - y1
    b14 = x1 - x4
    c14 = x4 * y1 - x1 * y4
    line14 = [a14,b14,c14]

    # l23，左下角到右上角
    a23 = y3 - y2
    b23 = x2 - x3
    c23 = x3 * y2 - x2 * y3
    line23 = [a23,b23,c23]

    return [line14, line23, line12, line34]


def calcY(x, param):
    """
    由x值计算y值
    """
    a = param[0]
    b = param[1]
    c = param[2]
    if b != 0:
        return int(-(c + a * x) / b)
    else:
        return 0


def cal_y_top_and_bottom(raw_img, position_pair, box):
    """
    计算每个anchor的上下边界
    """
    y_top = []
    y_bottom = []
    box = sortCoords(box)
    lines = calcLine(box)
    whichline = []
    for k in range(len(position_pair)):
        box_x = [box[0], box[2], box[4], box[6]]
        box_x.sort()
        box_left_left = box_x[0]
        box_left_right = box_x[1]
        box_right_left = box_x[2]
        box_right_right = box_x[3]
        anchor_middle_x = position_pair[k][0] + 7.5

        # 位于box中段的
        if anchor_middle_x >= box_left_right and anchor_middle_x <= box_right_left:
            anchor_y_top = calcY(anchor_middle_x, lines[0])
            anchor_y_bottom = calcY(anchor_middle_x, lines[1])
            y_top.append(anchor_y_top)
            y_bottom.append(anchor_y_bottom)
            whichline.append(1)
            continue
            elif anchor_middle_x > box_left_left and anchor_middle_x < box_left_right:  # 位于左边界上的
            if lines[2][1] == 0:
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                continue
            k_l12 = -(lines[2][0] / lines[2][1])
            anchor_y_top = calcY(anchor_middle_x, lines[0])
            anchor_y_bottom = calcY(anchor_middle_x, lines[1])
            if k_l12 > 0:
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[2])
            else:
                anchor_y_top = calcY(anchor_middle_x, lines[2])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
            y_top.append(anchor_y_top)
            y_bottom.append(anchor_y_bottom)
            whichline.append(2)
            continue

        elif anchor_middle_x <= box_left_left:
            anchor_middle_x = position_pair[k][1]  # 如果左边界外的，则用anchor右边缘替代中线
            if anchor_middle_x > box_right_right:  # 如果box很小时，anchor右边界超出box右边界，此时将anchor_middle_x替换为box_left_right
                anchor_middle_x = box_left_right
            if anchor_middle_x >= box_left_right and anchor_middle_x <= box_right_left:  # 位于box中段的
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                whichline.append(3)
                continue
            else:
                if lines[2][1] == 0:
                    anchor_y_top = calcY(anchor_middle_x, lines[0])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                    y_top.append(anchor_y_top)
                    y_bottom.append(anchor_y_bottom)
                    continue
                k_l12 = -(lines[2][0] / lines[2][1])
                if k_l12 > 0:
                    anchor_y_top = calcY(anchor_middle_x, lines[0])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[2])
                else:
                    anchor_y_top = calcY(anchor_middle_x, lines[2])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                whichline.append(4)
                continue

        elif anchor_middle_x > box_right_left and anchor_middle_x < box_right_right:  # 位于右边界上的
            if lines[3][1] == 0:
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                continue
            k_l34 = -(lines[3][0] / lines[3][1])
            if k_l34 > 0:
                anchor_y_top = calcY(anchor_middle_x, lines[3])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
            else:
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[3])
            y_top.append(anchor_y_top)
            y_bottom.append(anchor_y_bottom)
            whichline.append(5)
            continue

        elif anchor_middle_x >= box_right_right:
            anchor_middle_x = position_pair[k][0]  # anchor左边界替代中线
            if anchor_middle_x < box_left_left:  # 如果box很小时，anchor左边界超出box左边界，此时将anchor_middle_x替换为box_right_left
                anchor_middle_x = box_right_left
            if anchor_middle_x >= box_left_right and anchor_middle_x <= box_right_left:  # 位于box中段的
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                whichline.append(6)
                continue
            else:
                if lines[3][1] == 0:
                    anchor_y_top = calcY(anchor_middle_x, lines[0])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                    y_top.append(anchor_y_top)
                    y_bottom.append(anchor_y_bottom)
                    continue
                k_l34 = -(lines[3][0] / lines[3][1])
                if k_l34 > 0:
                    anchor_y_top = calcY(anchor_middle_x, lines[3])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                else:
                    anchor_y_top = calcY(anchor_middle_x, lines[0])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[3])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                whichline.append(7)
                continue

    # print(y_top)
    # print(y_bottom)
    # print(whichline)
    return y_top, y_bottom

def generate_gt_anchor(img, box, anchor_width=16, draw_img_gt=None):
    """
    将原先每张图的bbox标签转化为每个anchor标签
    :param img: 输入的图像
    :param box: bbox gt（8个数字表示一个框）
    :param anchor_width:anchor的宽度
    :param draw_img_gt:画出gt框后的图像
    :return:
    """
    if not isinstance(box[0], float):
        box = [float(box[i]) for i in range(len(box))]

    result = []
    # 求解一个bbox下，能分解为多少个16宽度的小anchor，并求出最左和最右的小achor的id
    left_anchor_num = int(math.floor(max(min(box[0], box[6]), 0) / anchor_width)) # float(x) 函数：对x向下取整
    right_anchor_num = int(math.ceil(min(max(box[2], box[4]), img.shape[1]) / anchor_width)) # ceil(x) 函数返回一个大于或等于 x 的的最小整数

    # 极端情况：右边的anchor超过了图像的宽度
    if right_anchor_num * 16 + 15 > img.shape[1]:
        right_anchor_num  -= 1

    # 将gt左右两边的anchor合并为一对
    position_pair = [(i * anchor_width, (i + 1) * anchor_width - 1) for i in range(left_anchor_num, right_anchor_num)]
    # 计算anchor的高度和中心是多少
    y_top, y_bottom = cal_y_top_and_bottom(img, position_pair, box)

    for i in range(len(position_pair)):
        position = int(position_pair[i][0] / anchor_width)  # anchor box的编号
        h = y_bottom[i] - y_top[i] + 1  # anchor box的高度
        cy = (float(y_bottom[i]) + float(y_top[i])) / 2.0  # anchor box的中心点
        result.append((position, cy, h))
        draw_img_gt = draw_image.draw_box_h_and_c(draw_img_gt, position, cy, h)
    draw_img_gt = draw_image.draw_box_4pt(draw_img_gt, box, color=(0, 0, 255), thickness=1)
    return result, draw_img_gt
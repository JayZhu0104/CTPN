import numpy as np
import math


def cal_y(cy, h):
    """
    根据cy、h计算出anchor的上界和下界
    """
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_bottom = int(cy + (float(h) - 1) / 2.0)
    return y_top, y_bottom


def valid_anchor(cy, h, height):
    """
    判断这个anchor是否有效
    """
    top, bottom = cal_y(cy, h)
    if top < 0:
        return False
    if bottom > (height * 16 - 1):
        return False
    return True


def cal_IoU(cy1, h1, cy2, h2):
    """
    计算IoU，因为x坐标都是一样的，所以只需要比较y坐标上
    前两个参数表示的是生成的anchor，后两个参数表示的是gt
    """
    y_top1, y_bottom1 = cal_y(cy1, h1)
    y_top2, y_bottom2 = cal_y(cy2, h2)
    y_top_min = min(y_top1, y_top2)
    y_bottom_max = max(y_bottom1, y_bottom2)
    union = y_bottom_max - y_top_min + 1
    intersection = h1 + h2 - union
    iou = float(intersection) / union
    if iou < 0:
        return 0.0
    else:
        return iou


def tag_anchor(gt_anchor, cnn_output, gt_box):
    """
    :param gt_anchor: gt转化为的anchor
    :param cnn_output: vgg16最后的特征图
    :param gt_box: gt真实坐标
    :return: 正样本、负样本、y坐标上的差值、偏移量的差值
    """
    anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273] # 高度是从11到273，每次÷0.7

    height = cnn_output.shape[2]
    width = cnn_output.shape[3]
    positive = []
    negative = []
    vertical_reg = []
    side_refinement_reg = []
    x_left_side = min(gt_box[0], gt_box[6])
    x_right_side = max(gt_box[2], gt_box[4])
    left_side = False
    right_side = False
    for a in gt_anchor:
        if a[0] >= int(width - 1):
            continue

        # 如果gt的最左边点是在某个anchor内，则表示有左侧边
        if x_left_side in range(a[0] * 16, (a[0] + 1) * 16):
            left_side = True
        else:
            left_side = False
        if x_right_side in range(a[0] * 16, (a[0] + 1) * 16):
            right_side = True
        else:
            right_side = False

        iou = np.zeros((height, len(anchor_height)))
        temp_positive = []
        for i in range(iou.shape[0]):
            for j in range(iou.shape[1]):
                if not valid_anchor((float(i) * 16.0 + 7.5), anchor_height[j], height):
                    continue
                iou[i][j] = cal_IoU((float(i) * 16.0 + 7.5), anchor_height[j], a[1], a[2])

                # 正样本
                if iou[i][j] > 0.7:
                    temp_positive.append((a[0], i, j, iou[i][j]))
                    # 如果左右两边的边界在这个anchor内的话，需要将其加入到偏移量中
                    if left_side:
                        o = (float(x_left_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0 # 边界占anchor的比例
                        side_refinement_reg.append((a[0], i, j, o))
                    if right_side:
                        o = (float(x_right_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                        side_refinement_reg.append((a[0], i, j, o))
                # 负样本
                if iou[i][j] < 0.5:
                    negative.append((a[0], i, j, iou[i][j]))

                if iou[i][j] < 0.5:
                    vc = (a[1] - (float(i) * 16.0 + 7.5)) / float(anchor_height[j])
                    vh = math.log10(float(a[2]) / float(anchor_height[j]))
                    vertical_reg.append((a[0], i, j, vc, vh, iou[i][j]))

        # 如果iou都不是正样本，就选出iou最大的作为正样本
        if len(temp_positive) == 0:
            max_position = np.where(iou == np.max(iou))
            temp_positive.append((a[0], max_position[0][0], max_position[1][0], np.max(iou))) # max_position[0][0]:最大值的x坐标；max_position[1][0]：最大值的y坐标

            if left_side:
                o = (float(x_left_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                side_refinement_reg.append((a[0], max_position[0][0], max_position[1][0], o))
            if right_side:
                o = (float(x_right_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                side_refinement_reg.append((a[0], max_position[0][0], max_position[1][0], o))

            if np.max(iou) <= 0.5:
                vc = (a[1] - (float(max_position[0][0]) * 16.0 + 7.5)) / float(anchor_height[max_position[1][0]])
                vh = math.log10(float(a[2]) / float(anchor_height[max_position[1][0]]))
                vertical_reg.append((a[0], max_position[0][0], max_position[1][0], vc, vh, np.max(iou)))
        positive += temp_positive
    return positive, negative, vertical_reg, side_refinement_reg


import cv2
import numpy as np
import os
import copy
import lib.dataset_handler
import torch
import lib.draw_image
import Net.net as Net
import lib.utils
import lib.nms
import math
import random
import shutil
import sys

anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]

IMG_ROOT = "../common/OCR_TEST"
TEST_RESULT = './test_result'
THRESH_HOLD = 0.7
NMS_THRESH = 0.3
NEIGHBOURS_MIN_DIST = 50 # 两个anchor距离小于该值时，把两者合并
MIN_ANCHOR_BATCH = 2
MODEL = './model/ctpn-msra_ali-9-end.model'

def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def gen_test_images(img_root, test_num=10):
    """
    生成测试图片集，返回测试图片的路径
    """
    img_list = os.listdir(img_root)
    if test_num > 0:
        random_list = random.sample(img_list, test_num)
    else:
        random_list = img_list
    test_pair = []
    for im in random_list:
        name, _ = os.path.splitext(im)
        im_path = os.path.join(img_root, im)
        test_pair.append(im_path)
    return test_pair


def get_anchor_h(anchor, v):
    """
    获取anchor的高
    """
    vc = v[int(anchor[7]), 0, int(anchor[5]), int(anchor[6])]
    vh = v[int(anchor[7]), 1, int(anchor[5]), int(anchor[6])]
    cya = anchor[5] * 16 + 7.5
    ha = anchor_height[int(anchor[7])]
    cy = vc * ha + cya
    h = math.pow(10, vh) * ha
    return h


def meet_v_iou(y1, y2, h1, h2):
    """
    判断两个anchor之间在y坐标上和尺寸上的相似性
    """
    def overlaps_v(y1, y2, h1, h2):
        return max(0, y2 - y1 + 1) / min(h1, h2)
    def size_similarity(h1, h2):
        return min(h1, h2) / max(h1, h2)

    return overlaps_v(y1, y2, h1, h2) >= 0.6 and size_similarity(h1, h2) >= 0.6


def get_successions(v, anchors=[]):
    """
    把符合条件的相邻anchor合并为一个，形成文本框
    """
    texts = []
    for i, anchor in enumerate(anchors):
        neighbours = []
        neighbours.append(i)
        center_x1 = (anchor[2] + anchor[0]) / 2
        h1 = get_anchor_h(anchor, v)
        # 找第i个anchor的邻边
        for j in range(0, len(anchors)):
            if j == i:
                continue
            center_x2 = (anchors[j][2] + anchors[j][0]) / 2
            h2 = get_anchor_h(anchors[j], v)
            if abs(center_x1 - center_x2) < NEIGHBOURS_MIN_DIST and \
                    meet_v_iou(max(anchor[1], anchors[j][1]), min(anchor[3], anchors[j][3]), h1, h2):
                neighbours.append(j)
        if len(neighbours) != 0:
            texts.append(neighbours)

    # 接下来把相邻的若干个anchor进行合并
    need_merge = True
    while need_merge:
        need_merge = False
        for i, line in enumerate(texts):
            if len(line) == 0:
                continue
            for index in line:
                # 如果texts[i]内有若一数字x，
                # 则遍历tests[i+1, len(texts)]，
                # 发现texts[j]内也有x，
                # 则texts[j]的内容应该与texts[i]合并，因为他们都属于同一个文本框
                for j in range(i+1, len(texts)):
                    if index in texts[j]:
                        texts[i] += texts[j]
                        texts[i] = list(set(texts[i]))
                        texts[j] = []
                        need_merge = True

    result = []
    for text in texts:
        if len(text < MIN_ANCHOR_BATCH):
            continue
        local = []
        for j in text:
            local.append(anchors[j])
        result.append(local)
    return result


def fit_y(X, Y, x1, x2):
    """
    通过X,Y计算出一条拟合直线f，再返回f(x1),f(x2)的结果
    """
    # 如果X只包括一个点，那么这个函数的拟合直线就是y=Y[0]
    if np.sum(X == X[0]) == len(X):
        return Y[0], Y[0]
    p = np.poly1d(np.polyfit(X, Y, 1)) # polyld就是拟合函数
    return p(x1), p(x2)


def clip_boxes(boxes, im_shape):
    """
    裁剪超过图像部分的box，
    """
    boxes[:, 0::2] = threshold(boxes[:, 0::2], 0, im_shape[1] - 1)
    boxes[:, 1::2] = threshold(boxes[:, 1::2], 0, im_shape[0] - 1)
    return boxes


def get_text_lines(text_proposals, im_size, scores=0):
    """
    对文本框的边缘做一个修整
    :param text_proposals: 若干个anchor合并形成的文本框
    :param im_size: 图片的大小
    :param scores:
    :return:
    """
    text_lines = np.zeros((len(text_proposals), 8), np.float32)

    for index, tp_indices in enumerate(text_proposals):
        text_line_boxes = np.array(tp_indices) # 每个文本框的小anchor
        X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2 # 求每一个小框的中心x，y坐标
        Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2

        z1 = np.polyfit(X, Y, 1) # 多项式拟合，根据之前求的中心店拟合一条直线,z1为多项式的各个系数

        x0 = np.min(text_line_boxes[:, 0])  # 文本行x坐标最小值
        x1 = np.max(text_line_boxes[:, 2])  # 文本行x坐标最大值

        offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5 # 小框宽度的一半

        # 要得到最后的文本框，还需要对其边界做一个拟合，看起来更自然
        # 以全部小框的左上角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
        lt_y, rt_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
        # 以全部小框的左下角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
        lb_y, rb_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

        text_lines[index, 0] = x0
        text_lines[index, 1] = min(lt_y, rt_y)  # 文本行上端 线段 的y坐标的小值
        text_lines[index, 2] = x1
        text_lines[index, 3] = max(lb_y, rb_y)  # 文本行下端 线段 的y坐标的大值
        text_lines[index, 4] = scores  # 文本行得分
        text_lines[index, 5] = z1[0]  # 根据中心点拟合的直线的k，b
        text_lines[index, 6] = z1[1]
        height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))  # 小框平均高度
        text_lines[index, 7] = height + 2.5

    text_recs = np.zeros((len(text_lines), 9), np.float32)
    index = 0
    for line in text_lines:
        b1 = line[6] - line[7] / 2  # 根据高度和文本行中心线，求取文本行上下两条线的b值
        b2 = line[6] + line[7] / 2
        x1 = line[0]
        y1 = line[5] * line[0] + b1  # 左上
        x2 = line[2]
        y2 = line[5] * line[2] + b1  # 右上
        x3 = line[0]
        y3 = line[5] * line[0] + b2  # 左下
        x4 = line[2]
        y4 = line[5] * line[2] + b2  # 右下
        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX * disX + disY * disY)  # 文本行宽度

        fTmp0 = y3 - y1 # 文本行高度
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1 * disX / width)  # 做补偿
        y = np.fabs(fTmp1 * disY / width)
        if line[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y
        # clock-wise order
        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x4
        text_recs[index, 5] = y4
        text_recs[index, 6] = x3
        text_recs[index, 7] = y3
        text_recs[index, 8] = line[4]
        index = index + 1

    text_recs = clip_boxes(text_recs, im_size)

    return text_recs

def infer_one(im_name, net):
    im = cv2.imread(im_name)
    im = lib.dataset_handler.scale_img_only(im)
    img = copy.deepcopy(im)
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]
    img = torch.Tensor(img)
    v, score, side = net(img, val=True)
    result = []
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            for k in range(score.shape[2]):
                if score[i, j, k, 1] > THRESH_HOLD:
                    result.append((j, k, i, float(score[i, j, k, 1].detach().numpy())))

    for_nms = []
    for box in result:
        pt = lib.utils.trans_to_2pt(box[1], box[0] * 16 + 7.5, anchor_height[box[2]])
        for_nms.append([pt[0], pt[1], pt[2], pt[3], box[3], box[0], box[1], box[2]])
    for_nms = np.array(for_nms, dtype=np.float32)
    nms_result = lib.nms.cpu_nms(for_nms, NMS_THRESH)

    out_nms = []
    for i in nms_result:
        out_nms.append(for_nms[i, 0:8])

    connect = get_successions(v, out_nms)
    texts = get_text_lines(connect, im.shape)

    for box in texts:
        box = np.array(box)
        print(box)
        lib.draw_image.draw_ploy_4pt(im, box[0:8], thickness=2)

    _, basename = os.path.split(im_name)
    cv2.imwrite('./infer_'+basename, im)

    for i in nms_result:
        vc = v[int(for_nms[i, 7]), 0, int(for_nms[i, 5]), int(for_nms[i, 6])]
        vh = v[int(for_nms[i, 7]), 1, int(for_nms[i, 5]), int(for_nms[i, 6])]
        cya = for_nms[i, 5] * 16 + 7.5
        ha = anchor_height[int(for_nms[i, 7])]
        cy = vc * ha + cya
        h = math.pow(10, vh) * ha
        lib.draw_image.draw_box_2pt(im, for_nms[i, 0:4])
    _, basename = os.path.split(im_name)
    cv2.imwrite('./infer_anchor_'+basename, im)

def random_test(net):
    test_pair = gen_test_images(IMG_ROOT, 0)
    print(test_pair)
    if os.path.exists(TEST_RESULT):
        shutil.rmtree(TEST_RESULT) # 删除一个目录以及目录内的所有内容

    os.mkdir(TEST_RESULT)

    for t in test_pair:
        im = cv2.imread(t)
        im = lib.dataset_handler.scale_img_only(im)
        img = copy.deepcopy(im)
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, :, :, :]
        img = torch.Tensor(img)
        v, score, side = net(img, val=True)
        result = []
        for i in range(score.shape[0]):
            for j in range(score.shape[1]):
                for k in range(score.shape[2]):
                    if score[i, j, k, 1] > THRESH_HOLD:
                        result.append((j, k, i, float(score[i, j, k, 1].detach().numpy()))) # detach:反向传播的时候，这个参数会更新，其他不会

        for_nms = []
        for box in result:
            pt = lib.utils.trans_to_2pt(box[1], box[0] * 16 + 7.5, anchor_height[box[2]])
            for_nms.append([pt[0], pt[1], pt[2], pt[3], box[3], box[0], box[1], box[2]])
        for_nms = np.array(for_nms, dtype=np.float32) # 这里保存的是需要进行nms操作的数据
        nms_result = lib.nms.cpu_nms(for_nms, NMS_THRESH)

        out_nms = []
        for i in nms_result:
            out_nms.append(for_nms[i, 0:8])

        connect = get_successions(v, out_nms)
        texts = get_text_lines(connect, im.shape)

        for i in nms_result:
            vc = v[int(for_nms[i, 7]), 0, int(for_nms[i, 5]), int(for_nms[i, 6])]
            vh = v[int(for_nms[i, 7]), 1, int(for_nms[i, 5]), int(for_nms[i, 6])]
            cya = for_nms[i, 5] * 16 + 7.5
            ha = anchor_height[int(for_nms[i, 7])]
            cy = vc * ha + cya
            h = math.pow(10, vh) * ha
            lib.draw_image.draw_box_2pt(im, for_nms[i, 0:4])

        for box in texts:
            box = np.array(box)
            lib.draw_image.draw_ploy_4pt(im, box[0:8], thickness=2)
        cv2.imwrite(os.path.join(TEST_RESULT, os.path.basename(t)), im)







if __name__ == '__main__':
    running_mode = sys.argv[2] # cpu还是gpu
    print("Mode: %s" % running_mode)
    net = Net.CTPN()

    if running_mode == 'cpu':
        net.load_state_dict(torch.load(MODEL, map_location=running_mode))
    else:
        net.load_state_dict(torch.load(MODEL))
    print(net)
    net.eval()

    if sys.argv[1] == 'random':
        random_test(net)
    else:
        url = sys.argv[1]
        infer_one(url, net)
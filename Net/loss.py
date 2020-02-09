import torch.nn as nn
import torch
import random

class CTPN_Loss(nn.Module):
    """
    损失函数：
    vertical_pred的regression loss用的是 SmoothL1Loss；
    score的classification loss，用的是CrossEntropyLoss；
    side refinement loss，用的是SmoothL1Loss。
    """
    def __init__(self, using_cuda=False):
        super(CTPN_Loss, self).__init__()
        # 根据损失函数的公式，定义一些固定参数
        self.Ns = 128 # score anchor的数量
        self.ratio = 0.5 # 正负样本比例
        self.lambda1 = 1.0 # 平衡因子
        self.lambda2 = 1.0 # 平衡因子
        # 三个损失函数的各自损失函数
        self.Ls_cls = nn.CrossEntropyLoss()
        self.Lv_reg = nn.SmoothL1Loss()
        self.Lo_reg = nn.SmoothL1Loss()

        self.using_cuda = using_cuda

    def forward(self, score, vertical_pred, side_refinement, positive, negative, vertical_reg, side_refinement_reg):
        """
        :param score: 预测分数
        :param vertical_pred: 预测的y坐标系数（anchor中心点y坐标，anchor高h）
        :param side_refinement: 预测的偏移量
        :param positive: gt的正样本
        :param negative: gt的负样本
        :param vertical_reg: gt的y坐标系数（anchor中心点y坐标，anchor高h）
        :param side_refinement_reg: gt的偏移量
        :return: loss
        """

        # 计算分类损失
        positive_num = min(int(self.Ns * self.ratio), len(positive))
        negative_num = self.Ns - positive_num
        positive_batch = random.sample(positive, positive_num) # 从positive中选择positive_num个随机且独立的元素
        negative_batch = random.sample(negative, negative_num)
        cls_loss = 0.0
        if self.using_cuda:
            for p in positive_batch:
                cls_loss += self.Ls_cls(score[0, p[2] * 2:((p[2] + 1) * 2), p[1], p[0]].unsqueeze(0), torch.LongTensor([1]).cuda())
            for n in negative_batch:
                cls_loss += self.Ls_cls(score[0, n[2] * 2:((n[2] + 1) * 2), n[1], n[0]].unsqueeze(0), torch.LongTensor([0]).cuda())
        else:
            for p in positive_batch:
                cls_loss += self.Ls_cls(score[0, p[2] * 2:((p[2] + 1) * 2), p[1], p[0]].unsqueeze(0),
                                        torch.LongTensor([1]))
            for n in negative_batch:
                cls_loss += self.Ls_cls(score[0, n[2] * 2:((n[2] + 1) * 2), n[1], n[0]].unsqueeze(0),
                                        torch.LongTensor([0]))

        cls_loss = cls_loss / self.Ns

        # 计算y坐标系相关系数损失
        v_reg_loss = 0.0
        Nv = len(vertical_reg)
        if self.using_cuda:
            for v in vertical_reg:
                v_reg_loss += self.Lv_reg(vertical_pred[0, v[2] * 2:((v[2] + 1) * 2), v[1], v[0]].unsqueeze(0), torch.FloatTensor([v[3], v[4]]).unsqueeze(0).cuda())
        else:
            for v in vertical_reg:
                v_reg_loss += self.Lv_reg(vertical_pred[0, v[2] * 2:((v[2] + 1) * 2), v[1], v[0]].unsqueeze(0),
                                          torch.FloatTensor([v[3], v[4]]).unsqueeze(0))
        v_reg_loss = v_reg_loss / float(Nv)

        # 计算偏移量的损失
        o_reg_loss = 0.0
        No = len(side_refinement_reg)
        if self.using_cuda:
            for s in side_refinement_reg:
                o_reg_loss += self.Lo_reg(side_refinement[0, s[2]:s[2] + 1, s[1], s[0]].unsqueeze(0), torch.FloatTensor([s[3]]).unsqueeze(0).cuda())
        else:
            for s in side_refinement_reg:
                o_reg_loss += self.Lo_reg(side_refinement[0, s[2]:s[2] + 1, s[1], s[0]].unsqueeze(0),
                                          torch.FloatTensor([s[3]]).unsqueeze(0))
        o_reg_loss = o_reg_loss / float(No)

        loss = cls_loss + v_reg_loss * self.lambda1 + o_reg_loss * self.lambda2

        return loss, cls_loss, v_reg_loss, o_reg_loss




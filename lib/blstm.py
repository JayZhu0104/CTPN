import torch
import torch.nn as nn
import numpy as np

class BLSTM(nn.Module):
    """
    BLSTM 网络层
    """
    def __init__(self, channel, hidden_unit, bidirectional=True):
        """
        :param channel: lstm 输入通道数
        :param hidden_unit:lstm 隐藏单元
        :param bidirectional:双向
        """
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(channel, hidden_unit, bidirectional=bidirectional)

    def forward(self, x):
        x = x.transpose(1, 3)
        recurrent, _ = self.lstm(x[0])
        recurrent = recurrent[np.newaxis, :, :, :]
        recurrent = recurrent.transpose(1, 3)
        return recurrent

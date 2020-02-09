import torch


def trans_to_2pt(position, cy, h, anchor_width=16):
    """
    通过anchor编号、cy、h转化成两个坐标点（左上，右下）来表示anchor
    """
    x_left = position * anchor_width
    x_right = (position + 1) * anchor_width - 1
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_bottom = int(cy + (float(h) - 1) / 2.0)
    return [x_left, y_top, x_right, y_bottom]

def init_weight(net):
    """
    初始化网络参数
    """
    for i in range(len(net.rnn.blstm.lstm.all_weights)):
        for j in range(len(net.rnn.blstm.lstm.all_weights[0])):
            torch.nn.init.normal_(net.rnn.blstm.lstm.all_weights[i][j], std=0.01)

    torch.nn.init.normal_(net.FC.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.FC.bias, val=0)

    torch.nn.init.normal_(net.vertical_coordinate.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.vertical_coordinate.bias, val=0)

    torch.nn.init.normal_(net.score.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.score.bias, val=0)

    torch.nn.init.normal_(net.side_refinement.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.side_refinement.bias, val=0)

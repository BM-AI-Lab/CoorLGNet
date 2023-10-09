import torch
from torch import nn
from torch.nn.parameter import Parameter
import math

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.gelu = nn.GELU()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        # y = self.gelu(y)

        return x * y.expand_as(x)   # 将y扩展成和x一样的维度



# class eca_layer(nn.Module):           # Efficient Channel Attention module
#     def __init__(self, channel, b=1, gamma=2):
#         super(eca_layer, self).__init__()
#         t = int(abs((math.log(channel, 2) + b) / gamma))
#         k = t if t % 2 else t + 1
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         out = self.avg_pool(x)
#         out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         out = self.sigmoid(out)
#         return x * out.expand_as(out)
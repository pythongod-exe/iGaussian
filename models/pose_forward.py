''' Spatial-Temporal Transformer Networks
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import ModelUtils
from .encoder_decoder import UNetx3

import pdb

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################


class SpatialAttention(nn.Module):
    """
    Compute 'Scaled Dot Product SpatialAttention
    """

    def forward(self, key, query, value):
        n, c, h, w = key.shape
        _, _, _, _ = query.shape
        _, c2, _, _ = value.shape

        key = key.reshape(n, c, -1).permute(0, 2, 1)
        query = query.reshape(n, c, -1)
        value = value.reshape(n, c2, -1)
        scores = torch.matmul(key, query
                              ) / math.sqrt(query.size(-2))
        p_attn = F.softmax(scores, dim=-2)
        # print(f'SpatAttn {value.shape=} {p_attn.shape=}')
        p_val = torch.matmul(value, p_attn)
        p_val = p_val.reshape(n, c2, h, w)
        return p_val


class AttnBlock(nn.Module):
    def __init__(self, d_model, kq_channels=3):
        super().__init__()
        self.query_embedding = UNetx3(kq_channels, kq_channels, use_norm=True)
        self.key_embedding = UNetx3(kq_channels, kq_channels, use_norm=True)
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0, bias=False)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))


class SpatialAttnBlock(AttnBlock):
    def __init__(self, d_model, kq_channels=3):
        super().__init__(d_model, kq_channels=kq_channels)
        self.attention = SpatialAttention()

    def forward(self, k, q, v):
        _value = self.value_embedding(v)
        v = self.attention(k, q, _value)

        return v
    def update_v(self, v_old, v):
        v = v_old + self.output_linear(v)
        return v


# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model, output_channels=None, use_bias=False):
        super(FeedForward, self).__init__()
        if output_channels is None:
            output_channels = d_model
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, output_channels, kernel_size=3, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x): # (batch_size, d_model, height, width)
        x = self.conv(x)
        return x


class SpatialTransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden=128, height=48, width=64, tmp_size=4):
        super().__init__()
        self.feed_forward = FeedForward(d_model=hidden, output_channels=hidden)

    def forward(self, concate_feature):
        feature = self.feed_forward(concate_feature)

        return feature

class CrossAttention(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden=128, height=48, width=64, tmp_size=4):
        super().__init__()
        # self.attention = SpatialAttnBlock(d_model=hidden, kq_channels=hidden * 3) # [1, 1, 1, 393216]
        self.attention = SpatialAttnBlock(d_model=hidden, kq_channels=hidden*3) 
        self.feed_forward = FeedForward(d_model=hidden, output_channels=hidden)

    def forward(self, k, q, v): # [1, 128, 2500, 1] 
        v2 = self.attention(k, q, v)

        v = self.attention.update_v(v, v2)
        v = v + self.feed_forward(v)

        return v

if __name__ == '__main__':
    pass

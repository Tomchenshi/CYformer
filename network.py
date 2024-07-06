import torch
import math
import torch.nn as nn
from common import *
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
import scipy.io as sio
from CITM import *


class Net(nn.Module):
    """
        Args:
            inp_channels (int, optional): Input channels of HSI. Defaults to 31.
            dim (int, optional): Embedding dimension. Defaults to 90.
            depths (list, optional): Number of Transformer block at different layers of network. Defaults to [ 6,6,6,6,6,6].
            num_heads (list, optional): Number of attention heads in different layers. Defaults to [ 6,6,6,6,6,6].
            mlp_ratio (int, optional): Ratio of mlp dim. Defaults to 2.
            qkv_bias (bool, optional): Learnable bias to query, key, value. Defaults to True.
            qk_scale (_type_, optional): The qk scale in non-local spatial attention. Defaults to None. If it is set to None, the embedding dimension is used to calculate the qk scale.
            bias (bool, optional):  Defaults to False.
            drop_path_rate (float, optional):  Stochastic depth rate of drop rate. Defaults to 0.1.
    """

    def __init__(self,
                 inp_channels=31,
                 dim=60,
                 depths=[1, 1, 1, 1, 1, 1],
                 num_heads=[6, 6, 6, 6, 6, 6],
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 bias=False,
                 drop_path_rate=0.1,
                 scale=4
                 ):
        super(Net, self).__init__()

        self.conv_first1 = nn.Conv2d(inp_channels, dim, 3, 1, 1)  # shallow featrure extraction
        self.conv_first2 = nn.Conv2d(3, dim, 3, 1, 1)  # shallow featrure extraction
        self.num_layers = depths
        self.hsi_module = nn.ModuleList()
        print("network depth:", len(self.num_layers))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i_layer in range(len(self.num_layers)):
            layer1 = CITM(dim=dim,
                             window_size=8,
                             depth=depths[i_layer],
                             num_head=num_heads[i_layer],
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                             bias=bias,
                             n_scale=scale)

            self.hsi_module.append(layer1)

        # self.conv_delasta = nn.Conv2d(dim, inp_channels, 3, 1, 1)  # reconstruction from features
        self.skip_conv = default_conv(inp_channels, dim, 3)
        self.upsample = Upsampler(default_conv, scale, dim)
        self.tail = default_conv(dim*3, inp_channels, 3)
        self.conv = default_conv(2 * dim,dim,1)
        self.high1 = Updownblock()
        self.bicubicsample = torch.nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.sc = default_conv(dim, dim, 3)


    def forward(self, lr_hsi, rgb):
        hr_hsi = self.bicubicsample(lr_hsi)
        x1 = self.conv_first1(hr_hsi)
        x2 = self.conv_first2(rgb)
        x = torch.cat([x1, x2], dim=1)
        x2 = self.dw(x2)
        x1 = self.sc(x1)
        x = self.conv(x)
        for i_layer in range(len(self.num_layers)):
            x = self.hsi_module[i_layer](x, x2, x1)
        #x = x + x1 + x2
        x = torch.cat([x, x2, x1], dim=1)
        x = self.tail(x)
        return x


class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class Updownblock(nn.Module):
    def __init__(self):
        super(Updownblock, self).__init__()
        self.down = nn.AvgPool2d(kernel_size=2)
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.down(x)
        high = x - F.interpolate(x1, size = x.size()[-2:], mode='bilinear', align_corners=True)
        return self.act(high)


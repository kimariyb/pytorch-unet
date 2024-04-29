# -*- conding: utf-8 -*-
"""
unet.py

U-Net模型定义

@author:
Kimariyb (kimariyb@163.com)

@license:
Licensed under the MIT License
For details, see the License file.

@Data:
2024/4/27 21:55
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # 第一次卷积操作，由于 stride=1，所以输入输出尺寸不变
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 二次卷积操作
            nn.Conv2d(
                out_channels, 
                out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)
        

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            # 下采样路径 (Encoder)
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        for feature in reversed(features):
            # 上采样路径 (Decoder)
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, 
                    feature, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # 最终的卷积层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            # 下采样路径 (Encoder) 的前向传播
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 瓶颈层的前向传播
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                # 对 x 进行调整大小，使其与对应的 skip_connection 形状相同
                x = F.interpolate(x, size=skip_connection.shape[2:])
            
             # 将 skip_connection与 x 进行连接，并进行上采样
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        # 最终卷积层的前向传播
        return self.final_conv(x)
    


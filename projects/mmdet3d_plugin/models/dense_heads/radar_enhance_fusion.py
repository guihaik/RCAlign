import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_attention
from mmcv.cnn import build_norm_layer
from mmdet.models import HEADS

@HEADS.register_module()
class RadarEnhanceFusion(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, padding=1):
        super(RadarEnhanceFusion, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        )

        self.conv_mid = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # self.query_linear = nn.Linear(out_channels, out_channels)
        # self.key_linear = nn.Linear(out_channels,out_channels)
        # self.value_linear = nn.Linear(out_channels, out_channels)
        # self.attention = build_attention(attention)
        # self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, radar, x):

        x = torch.cat([radar, x], dim=1)

        x = self.conv_in(x)
        x = self.conv_mid(x)
        x = self.conv_out(x)

        return x

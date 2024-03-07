import torch
import torch.nn as nn
import torch.nn.functional as F
from . vit_seg_modeling_resnet_skip import ResNetV2
from . coefficient_regression import coefficient
from . center_regression_net import center_regression


class network(nn.Module):
    def __init__(self, in_channel=3, out_channel=2, z=64, y=112, x=96, num_coefficient=500, args=None):
        super(network, self).__init__()
        self.hybrid_model = ResNetV2(block_units=(2, 3, 3), width_factor=1)
        self.width = 32
        self.dim = self.width * 8
        self.decoder_channels = (self.width * 4, self.width * 2, self.width * 1)
        self.skip_channels = [self.width * 4, self.width * 2, self.width * 1]
        channels = 16
        
        self.encoder1 = nn.Sequential(
            Conv3dReLU(in_channel, channels, kernel_size=3, padding=1),
            Conv3dReLU(channels, channels, kernel_size=1, padding=0)
        )
        
        self.decoder1 = nn.Sequential(
            Conv3dReLU(self.dim + self.skip_channels[0], self.decoder_channels[0], kernel_size=3, padding=1),
            Conv3dReLU(self.decoder_channels[0], self.decoder_channels[0], kernel_size=1, padding=0)
        )
        self.decoder2 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[0] + self.skip_channels[1], self.decoder_channels[1], kernel_size=3, padding=1),
            Conv3dReLU(self.decoder_channels[1], self.decoder_channels[1], kernel_size=1, padding=0)
        )
        self.decoder3 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[1] + self.skip_channels[2], self.decoder_channels[2], kernel_size=3, padding=1),
            Conv3dReLU(self.decoder_channels[2], self.decoder_channels[2], kernel_size=1, padding=0)
        )
        self.decoder4 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[2] + channels, channels, kernel_size=3, padding=1),
            Conv3dReLU(channels, channels, kernel_size=1, padding=0)
        )
        self.num_coefficient = num_coefficient
        self.coefficient = coefficient(num_coefficient=num_coefficient)
        self.center_regression = center_regression(z, y, x, args)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.segmentation_head = nn.Conv3d(channels, out_channel, kernel_size=3, padding=1)


    def forward(self, x):
        # print(x.shape)
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1,1)
        t1 = self.encoder1(x)
    
        x, features = self.hybrid_model(t1)
        
        deep_feature = x
        
        x = self.up(x)
        x = torch.cat((x, features[0]), 1)
        x = self.decoder1(x)

        
        x = self.up(x)
        x = torch.cat((x, features[1]), 1)
        x = self.decoder2(x)

        
        x = self.up(x)
        x = torch.cat((x, features[2]), 1)
        x = self.decoder3(x)

        
        x = self.up(x)
        x = torch.cat((x, t1), 1)
        x = self.decoder4(x)
        
        coefficient = self.coefficient(deep_feature, x)
        coefficient = coefficient[:, 0: self.num_coefficient]
        mask = self.segmentation_head(x)
        center = self.center_regression(deep_feature, mask[:, 1])
        return mask, center, coefficient



class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


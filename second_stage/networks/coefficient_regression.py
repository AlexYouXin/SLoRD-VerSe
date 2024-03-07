import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage




class coefficient(nn.Module):
    def __init__(self, num_coefficient=500):
        super(coefficient, self).__init__()
        self.width = 32
        self.dim = self.width * 8
        self.decoder_channels = (self.width * 4, self.width * 2, self.width * 1)
        self.skip_channels = [self.width * 4, self.width * 2, self.width * 1]
        channels = 16
        
        
        self.encoder1 = nn.Sequential(
            Conv3dReLU(self.dim, self.dim, kernel_size=3, padding=1),
            Conv3dReLU(self.dim, self.dim, kernel_size=1, padding=0)
        )
        self.encoder2 = nn.Sequential(
            Conv3dReLU(self.dim, self.dim, kernel_size=3, padding=1),
            Conv3dReLU(self.dim, self.dim, kernel_size=1, padding=0)
        )
        
        self.encoder3 = nn.Sequential(
            Conv3dReLU(self.dim + channels, self.dim + channels, kernel_size=3, padding=1),
            Conv3dReLU(self.dim + channels, num_coefficient, kernel_size=1, padding=0)
        )

        self.encoder4 = Conv3dbn(num_coefficient, num_coefficient, kernel_size=3, padding=1)


        
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x, mask_embedding):

        x = self.encoder1(x)
        x = self.encoder2(x)
        x = F.interpolate(x, scale_factor=4, mode="trilinear")
        
        mask_embedding = F.interpolate(mask_embedding, scale_factor=0.25, mode="trilinear")   # 16

        
        x = torch.cat((x, mask_embedding), 1)
        
        x = self.encoder4(self.encoder3(x))
        
        coefficient = self.pool(x).flatten(1)
        
        
        return coefficient


    


class Conv3dbn(nn.Sequential):
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

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dbn, self).__init__(conv, bn)



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

        

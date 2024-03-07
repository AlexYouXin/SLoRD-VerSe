import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


class center_regression(nn.Module):
    def __init__(self, z, y, x, args):
        super(center_regression, self).__init__()
        self.width = 32
        self.dim = self.width * 8
        self.decoder_channels = (self.width * 4, self.width * 2, self.width * 1)

        channels = 16

        self.encoder1 = nn.Sequential(
            Conv3dReLU(self.dim, self.decoder_channels[0], kernel_size=3, padding=1),
            Conv3dReLU(self.decoder_channels[0], self.decoder_channels[0], kernel_size=1, padding=0)
        )
        self.encoder2 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[0], self.decoder_channels[1], kernel_size=3, padding=1),
            Conv3dReLU(self.decoder_channels[1], self.decoder_channels[1], kernel_size=1, padding=0)
        )
        self.encoder3 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[1], self.decoder_channels[2], kernel_size=3, padding=1),
            Conv3dReLU(self.decoder_channels[2], self.decoder_channels[2], kernel_size=1, padding=0)
        )
        self.encoder4 = Conv3dbn(self.decoder_channels[2], 3, kernel_size=3, padding=1)

        z_index = torch.zeros((z, y, x))
        y_index = torch.zeros((z, y, x))
        x_index = torch.zeros((z, y, x))

        for i in range(z):
            z_index[i, :, :] = i

        for i in range(y):
            y_index[:, i, :] = i

        for i in range(x):
            x_index[:, :, i] = i
        self.num_batch = args.batch_size
        self.z_index = nn.Parameter(z_index, requires_grad=False)
        self.y_index = nn.Parameter(y_index, requires_grad=False)
        self.x_index = nn.Parameter(x_index, requires_grad=False)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x, mask):
        # x: B * C * dim * H/16 * W/16 * L/16
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        mask = torch.sigmoid(mask)
        center_residual = self.pool(x).flatten(1)  # b * 3
        # print(mask.shape, self.num_batch)
        z_center = torch.mean(mask * self.z_index.unsqueeze(0).repeat(self.num_batch, 1, 1, 1),
                              dim=(1, 2, 3)).unsqueeze(1)
        y_center = torch.mean(mask * self.y_index.unsqueeze(0).repeat(self.num_batch, 1, 1, 1),
                              dim=(1, 2, 3)).unsqueeze(1)
        x_center = torch.mean(mask * self.x_index.unsqueeze(0).repeat(self.num_batch, 1, 1, 1),
                              dim=(1, 2, 3)).unsqueeze(1)
        center = torch.cat((z_center, y_center, x_center), 1)
        # print(center.shape, center_residual.shape)
        center = center + center_residual
        return center


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

        
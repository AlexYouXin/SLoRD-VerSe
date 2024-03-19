import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import time
from skimage.morphology import convex_hull_image
import math


def spherical_to_cartesian(radius, theta, phi):
    theta = torch.deg2rad(theta)
    phi = torch.deg2rad(phi)

    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)

    return x, y, z

# convert spherical coordinate to cartesian coordinate
def decoding_theta(data, dim, step):
    centers = data['center']
    '''
    bboxs = np.array(list(data['bbox'])).astype(np.float32)
    x, y, z, w, h, d = bboxs

    bboxs_x1 = x - w / 2  # 1
    bboxs_x2 = x + w / 2  # 1
    bboxs_y1 = y - h / 2  # 1
    bboxs_y2 = y + h / 2  # 1
    bboxs_z1 = z - d / 2  # 1
    bboxs_z2 = z + d / 2  # 1
    bboxsw = np.abs(bboxs_x2 - bboxs_x1)  # 1
    bboxsh = np.abs(bboxs_y2 - bboxs_y1)  # 1
    bboxsd = np.abs(bboxs_z2 - bboxs_z1)  # 1
    relative_lens = np.sqrt(bboxsw * bboxsw + bboxsh * bboxsh + bboxsd * bboxsd)  # 1
    '''
    
    center_xs = centers[0].unsqueeze(2).type(torch.float32)  # 16 * 1
    center_ys = centers[1].unsqueeze(2).type(torch.float32)  # 16 * 1
    center_zs = centers[2].unsqueeze(2).type(torch.float32)  # 16 * 1

    idx = torch.linspace(0, 360 - step, dim[0], requires_grad=False).type(torch.float32).reshape(dim[0], 1).cuda()
    idy = torch.linspace(0, 180, dim[1], requires_grad=False).type(torch.float32).reshape(dim[1], 1).cuda()

    rs = data['r']
    # print(rs.shape)   # B * 72 * 37

    rs = rs.type(torch.float32)

    # idx_list = np.flip(idx, axis=0).astype(np.float32)
    # idy_list = np.flip(idy, axis=0).astype(np.float32)
    phi_list = idx * torch.ones((1, dim[1])).type(torch.float32).cuda()   # 72 * 37
    theta_list = idy.T * torch.ones((dim[0], 1)).type(torch.float32).cuda()    # 72 * 37

    y, x, z = spherical_to_cartesian(rs, theta_list, phi_list)
    # x, y = cv2.polarToCart(rs, theta_list, angleInDegrees=True)  # 360    360

    x1 = x + center_xs  # B * 72 * 37
    y1 = y + center_ys  # B * 72 * 37
    z1 = z + center_zs
    r1 = x**2 + y**2 + z**2

    polygons = torch.stack((x1, y1, z1), dim=1)              # B * 3 * 72 * 37

    return polygons, x1, y1, z1, r1



class contour_loss_module(nn.Module):
    def __init__(self, num_contour, base_num, args):
        super(contour_loss_module, self).__init__()
        self.num_contour = num_contour
        self.base_num = base_num
        self.scale = [20, 20, 20]
        self.h = args.img_size[0] / 2
        self.w = args.img_size[1] / 2
        self.l = args.img_size[2] / 2


        self.max_dist = math.sqrt(self.scale[0] / 2 * self.scale[0] / 2 + self.scale[1] / 2 * self.scale[1] / 2 + self.scale[2] / 2 * self.scale[2] / 2)
        self.loss = nn.L1Loss()
        self.verse = 5
        self.dim = [np.int(360 / self.verse), np.int(180 / self.verse + 1)]
        self.z = args.img_size[0]
        self.y = args.img_size[1]
        self.x = args.img_size[2]
        self.data = dict()


    def forward(self, coefficient, contour_gt, V, center):
        # print(coefficient.shape, contour_gt.shape, V.shape, center.shape)
        # torch.Size([16, 512]) torch.Size([16, 2664]) torch.Size([2664, 2664]) torch.Size([16, 3])
        base_vector_array = V[0: self.base_num, :]
        base_vector = torch.matmul(coefficient, base_vector_array)   # 16 * 2664

        center_z = center[:, 0].unsqueeze(1) * self.h
        center_y = center[:, 1].unsqueeze(1) * self.w
        center_x = center[:, 2].unsqueeze(1) * self.l
        # self.data['bbox'] = (center_z, center_y, center_x)
        self.data['r'] = base_vector
        self.data['center'] = (center_x, center_y, center_z)

        contour_vector = distance2mask(self.data, self.dim, self.verse, self.z, self.y, self.x)           # B * 2664 * 3

        # torch.cdist

        # print(contour_vector.device, contour_gt.device)            # cpu cuda:0
        # print(contour_vector.shape, contour_gt.shape)          # torch.Size([16, 2664, 3]) torch.Size([16, 4000, 3])
        co_dist = torch.cdist(contour_vector, contour_gt, p=1.0)         # B * 2664 * 10000
        co_dist = torch.min(co_dist, dim=2)[0]                # B * 2664

        # loss = self.loss(base_vector, contour_gt)
        distance_loss = torch.mean(torch.mean(co_dist, dim=1))
        return distance_loss



def distance2mask(data, dim, step, z, y, x):
    # z, y, x = args.img_size[0], args.img_size[1], args.img_size[2]
    B = data['r'].shape[0]
    contour_combine = torch.zeros((B, dim[0] * dim[1], 3)).cuda()
    data['r'] = data['r'].reshape(B, dim[0], dim[1])           # B * 2664 -> B * 72 * 37
    out, xr, yr, zr, r1 = decoding_theta(data, dim, step)         # B * 3 * 72 * 37
    # Transform back：0-z, 1-y, 2-x
    contour_combine[:, :, 2] = out[:, 0].flatten(1).clamp(0, x)
    # contour_combine[:, :, 2] = contour_combine[:, :, 2]
    contour_combine[:, :, 1] = out[:, 1].flatten(1).clamp(0, y)
    contour_combine[:, :, 0] = out[:, 2].flatten(1).clamp(0, z)

    return contour_combine

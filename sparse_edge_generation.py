
import torchvision
from torchvision.transforms import transforms as T
import torch
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.cm as mpl_color_map
import copy
import time
import matplotlib.pyplot as plt
from scipy import ndimage
import SimpleITK as sitk
import seaborn as sns
import argparse
import math
import random
import cv2
from utils import *
from center import *
# from center_compute import *
import vtkmodules.all as vtk
import vtkmodules.util.numpy_support as vtk_np
from mayavi import mlab
from center_to_contour_distance import *
import csv
import json
from skimage.segmentation import find_boundaries

def canny_operator(data_nii):
    origin = data_nii.GetOrigin()
    spacing = data_nii.GetSpacing()
    direction = data_nii.GetDirection()
    # data_nii = label_erase(data_nii, origin, spacing, direction)
    # change data type before edge detection
    data_float_nii = sitk.Cast(data_nii, sitk.sitkFloat32)

    canny_op = sitk.CannyEdgeDetectionImageFilter()
    canny_op.SetLowerThreshold(0.50)

    canny_op.SetUpperThreshold(1.0)
    canny_op.SetVariance(3)  # 1
    canny_op.SetMaximumError(0.5)
    canny_sitk = canny_op.Execute(data_float_nii)
    canny_sitk = sitk.Cast(canny_sitk, sitk.sitkInt16)
    canny_sitk.SetOrigin(origin)
    canny_sitk.SetSpacing(spacing)
    canny_sitk.SetDirection(direction)
    return canny_sitk



def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:

            Filelist.append(filename)
    return Filelist


def edge_generation(label, origin, spacing, direction):
    label[label >= 0.5] = 1
    label[label < 0.5] = 0
    res_mask = sitk.GetImageFromArray(label)
    res_mask.SetOrigin(origin)
    res_mask.SetSpacing(spacing)
    res_mask.SetDirection(direction)

    # dilation
    bm = sitk.BinaryErodeImageFilter()
    bm.SetKernelType(sitk.sitkBall)
    bm.SetKernelRadius(1)
    bm.SetForegroundValue(1)
    res_mask = bm.Execute(res_mask)
    res_mask = sitk.GetArrayFromImage(res_mask)
    res_mask = label - res_mask
    return res_mask


if __name__ == "__main__":
    img_folder = "xxxxxxxxx/label/"
    out_folder = 'xxxxxxxx/sparse_edge/'

    edge_index = np.zeros
    Filelist = get_filelist(img_folder)
    print(len(Filelist))

    max_nonzero_index = 0
    terminal = 3
    mean_index_num = 0


    for img_file_name in Filelist:
        data_dir = img_folder + img_file_name
        data_nii = sitk.ReadImage(data_dir)
        origin = data_nii.GetOrigin()
        spacing = data_nii.GetSpacing()
        direction = data_nii.GetDirection()
        label_array = sitk.GetArrayFromImage(data_nii)

        unique_labels = sorted(list(set(label_array[label_array != 0])))
        edge_array = np.zeros_like(label_array)

        for i in unique_labels:
            # Boundary GT
            # canny_sitk = canny_operator(data_nii)
            array_i = np.array(label_array == i).astype(int)
            edge_array_gt = edge_generation(array_i, origin, spacing, direction)
            # canny_sitk = canny_operator(data_nii)

            edge_array[edge_array_gt != 0] = i




        index = np.nonzero(edge_array)
        index_num = np.count_nonzero(edge_array)
        index = np.transpose(index)
        center = np.mean(index, axis=0)
        center = center.flatten().tolist()
        # print(center.shape)
        index = index[0: -1: terminal, :]
        index_array = np.ones((np.int(20000 / terminal), 3)) * 1000


        if index_num > max_nonzero_index:
            max_nonzero_index = index_num
        mean_index_num += index_num
        print('num of non-zero voxels: ', index_num)
        sparse_edge = np.zeros_like(edge_array)
        for i in range(index.shape[0]):
            sparse_edge[index[i][0], index[i][1], index[i][2]] = 1
        sparse_edge = sparse_edge * unique_labels[0]
        edge_sitk = sitk.GetImageFromArray(sparse_edge)
        edge_sitk.SetOrigin(origin)
        edge_sitk.SetSpacing(spacing)
        edge_sitk.SetDirection(direction)
        '''
        canny_sitk = sitk.GetArrayFromImage(canny_sitk)
        canny_sitk[canny_sitk == 1] = 8
        canny_sitk = sitk.GetImageFromArray(canny_sitk)
        '''

        sitk.WriteImage(edge_sitk, out_folder + img_file_name)
        print(img_file_name)

    print('max nonzero index: ', max_nonzero_index)
    print('mean nonzero index: ', mean_index_num)









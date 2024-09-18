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
import argparse
import math
import random
import cv2
from utils import *
from center import *
# from center_compute import *
from center_to_contour_distance import *
import csv
import json


parser = argparse.ArgumentParser()

parser.add_argument('--num_classes', type=int,
                    default=26, help='output channel of network')
parser.add_argument('--img_size', type=int, default=[64, 112, 64], help='input patch size of network input')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-2, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True


def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(filename)
    return Filelist



def combining_coefficient(points, dim, V):
    points_array = np.array(points)
    print('array shape', points_array.shape)
    points = np.array(points).reshape(1, dim[0] * dim[1])
    combine_parameter = np.dot(points, V.T)
    print('number of base vector: ', combine_parameter.shape)  # number of base vector:  (1, 2664)
    return combine_parameter


def patch_crop(args, image, label, center):
    min_value = np.min(image)
    z = center[0]
    y = center[1]
    x = center[2]



    random_z = random.randint(-4, 5)
    random_y = random.randint(-4, 5)
    random_x = random.randint(-4, 5)

    crop_z_down = np.int(z) - np.int(args.img_size[0] / 2) + random_z
    crop_z_up = np.int(z) + np.int(args.img_size[0] / 2) + random_z
    crop_y_down = np.int(y) - np.int(args.img_size[1] / 2) + random_y
    crop_y_up = np.int(y) + np.int(args.img_size[1] / 2) + random_y
    crop_x_down = np.int(x) - np.int(args.img_size[2] / 2) + random_x
    crop_x_up = np.int(x) + np.int(args.img_size[2] / 2) + random_x


    if crop_z_down < 0 or crop_z_up > image.shape[0]:
        delta_z = np.int(np.maximum(np.abs(crop_z_down), np.abs(crop_z_up - image.shape[0])))
        # print('delta_z: ', delta_z)
        image = np.pad(image, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=min_value)
        label = np.pad(label, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=0.0)

        crop_z_down = crop_z_down + delta_z
        crop_z_up = crop_z_up + delta_z

    if crop_y_down < 0 or crop_y_up > image.shape[1]:
        delta_y = np.int(np.maximum(np.abs(crop_y_down), np.abs(crop_y_up - image.shape[1])))
        # print('delta_y:', delta_y)
        image = np.pad(image, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=min_value)
        label = np.pad(label, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=0.0)

        crop_y_down = crop_y_down + delta_y
        crop_y_up = crop_y_up + delta_y

    if crop_x_down < 0 or crop_x_up > image.shape[2]:
        delta_x = np.int(np.maximum(np.abs(crop_x_down), np.abs(crop_x_up - image.shape[2])))
        # print('delta_x:', delta_x)
        image = np.pad(image, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=min_value)
        label = np.pad(label, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=0.0)

        crop_x_down = crop_x_down + delta_x
        crop_x_up = crop_x_up + delta_x

    crop_z_down = np.int(crop_z_down)
    crop_z_up = np.int(crop_z_up)
    crop_y_down = np.int(crop_y_down)
    crop_y_up = np.int(crop_y_up)
    crop_x_down = np.int(crop_x_down)
    crop_x_up = np.int(crop_x_up)

    print('index: ', crop_z_down, crop_z_up, crop_y_down, crop_y_up, crop_x_down, crop_x_up)
    label_crop = label[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
    image_crop = image[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]

    return image_crop, label_crop, random_z, random_y, random_x



def msd_loss(pred, label):
    loss = np.sqrt(np.sum((pred - label) ** 2))
    return loss


def svd_restore(mat, U, S, V):
    n, l = mat.shape
    k = np.minimum(n, l)
    S_restore = np.zeros_like(mat)
    for i in range(k):
        S_restore[i, i] = S[i]
    return np.dot(np.dot(U, S_restore), V)


def singular_value(S, k):
    num = len(S)
    # k = 350
    ratio = np.sum(S[0: k]) / np.sum(S)
    return k, ratio


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":

    root_folder = 'xxxxxx/medical data/verse19_resample/'


    img_folder = root_folder + 'img/'
    label_folder = root_folder + 'label/'

    width = 40
    height = 30
    z = 20
    max_dist = math.sqrt(width / 2 * width / 2 + height / 2 * height / 2 + z / 2 * z / 2)

    # SVD
    mat = np.load('xxxxx/verse19_distance_final_5.npy')
    mat = mat / max_dist

    print('mat shape: ', mat.shape)
    n, r, c = mat.shape

    mat = np.reshape(mat, (n, -1))

    n, l = mat.shape

    k = np.minimum(n, l)
    # svd
    U, S, V = np.linalg.svd(mat)
    print('U shape: ', U.shape)
    print('S shape: ', S.shape)
    print('V shape: ', V.shape)
    # U shape:  (760, 760)
    # S shape:  (760,)
    # V shape:  (2664, 2664)
    # print('S: ', S)


    mat_restore = svd_restore(mat, U, S, V)
    error = msd_loss(mat_restore, mat)
    print('error: ', error)

    # singular value
    number_base = 400
    num_base, ratio = singular_value(S, number_base)
    print('ratio of singular value: ', ratio)

    verse = 5
    dim = (int(360/verse), int(180/verse + 1))
    print(dim)

    Filelist = get_filelist(img_folder)
    print(len(Filelist))

    for img_file_name in Filelist:

        img_file_path = os.path.join(img_folder, img_file_name)
        label_file_path = os.path.join(label_folder, img_file_name)
        img = sitk.ReadImage(img_file_path)
        label = sitk.ReadImage(label_file_path)
        origin = img.GetOrigin()
        direction = img.GetDirection()
        space = img.GetSpacing()
        img_array = sitk.GetArrayFromImage(img)
        label_array = sitk.GetArrayFromImage(label)

        space_array = np.array(space)


        if space_array[2] == 1.0:
            print(img_file_name)
            unique_labels = sorted(list(set(label_array[label_array != 0])))
            print(unique_labels, len(unique_labels))

            with open('name.txt', 'a') as file1, open('output.csv', 'a', newline='') as file2:  # 'a'
                for label_index in unique_labels:
                    
                    img_mask = (label_array == label_index).astype(np.uint8)
                    points, center_x, center_y, center_z = getOrientedPoints(img_mask, verse, dim)
                    print('center: ', center_z, center_y, center_x)
                    
                    if center_x == 0 and center_y == 0 and center_z == 0:
                        continue
                    print('size of points: ', len(points), len(points[0]))  # size of points:  72 37
                    combine_parameter = combining_coefficient(points, dim, V)
                    combine_parameter = combine_parameter.flatten().tolist()


                    file1.write(img_file_name.replace('_ct.nii.gz', '-{}_ct.nii.gz').format(label_index) + '\n')

                    center_list = []
                    center_list.append(center_z)
                    center_list.append(center_y)
                    center_list.append(center_x)
                    image_crop, label_crop, random_z, random_y, random_x = patch_crop(args, img_array, label_array, center_list)
                    center = []
                    center.append(np.int(args.img_size[0] / 2) - random_z)
                    center.append(np.int(args.img_size[1] / 2) - random_y)
                    center.append(np.int(args.img_size[2] / 2) - random_x)

                    center = str(center)
    
                    combine_parameter = str(combine_parameter)
    
                    vocabulary = []
                    vocabulary.append(img_file_name.replace('_ct.nii.gz', '-{}_ct.nii.gz').format(label_index))
                    vocabulary.append(center)
                    vocabulary.append(combine_parameter)


                    # jsonstr = json.dumps(vocabulary, cls=NumpyEncoder)
                    # file2.write(jsonstr)
                    # file2.write('\n')

                    csv_writer = csv.writer(file2)
                    csv_writer.writerow(vocabulary)


                    label_ = np.zeros_like(label_crop).copy()
                    label_[label_crop == label_index] = label_index

                    image_crop = sitk.GetImageFromArray(image_crop)
                    label_ = sitk.GetImageFromArray(label_)
    
                    image_crop.SetOrigin(origin)
                    image_crop.SetDirection(direction)
                    image_crop.SetSpacing(space)
                    label_.SetOrigin(origin)
                    label_.SetDirection(direction)
                    label_.SetSpacing(space)

                    sitk.WriteImage(image_crop, 'xxxxxxx/single_label/img/' + img_file_name.replace('_ct.nii.gz', '-{}_ct.nii.gz').format(label_index))
                    sitk.WriteImage(label_, 'xxxxxxx/single_label/label/' + img_file_name.replace('_ct.nii.gz', '-{}_ct.nii.gz').format(label_index))
                    print('label index: ', label_index)

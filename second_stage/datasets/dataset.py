import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
import argparse
# from torchvision import transforms
from torch.utils.data import DataLoader
import SimpleITK as sitk
import csv


def random_rot_flip(image, label):
    # k--> angle
    # i, j: axis
    k = np.random.randint(0, 4)
    axis = random.sample(range(0, 3), 2)
    image = np.rot90(image, k, axes=(axis[0], axis[1]))  # rot along z axis
    label = np.rot90(label, k, axes=(axis[0], axis[1]))

    # axis = np.random.randint(0, 2)
    # image = np.flip(image, axis=axis)
    # label = np.flip(label, axis=axis)
    flip_id = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
    image = np.ascontiguousarray(image[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    label = np.ascontiguousarray(label[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    return image, label


def random_rotate(image, label, min_value):
    angle = np.random.randint(-15, 15)  # -20--20
    rotate_axes = [(0, 1), (1, 2), (0, 2)]
    k = np.random.randint(0, 3)
    # image = ndimage.rotate(image, angle, reshape=False, order=3, mode='constant', cval=-2.0)
    image = ndimage.interpolation.rotate(image, angle, axes=rotate_axes[k], reshape=False, order=3, mode='constant',
                                         cval=min_value)
    # label = ndimage.rotate(label, angle, reshape=False, order=0, mode='constant', cval=0.0)
    label = ndimage.interpolation.rotate(label, angle, axes=rotate_axes[k], reshape=False, order=0, mode='constant',
                                         cval=0.0)
    # edge = ndimage.interpolation.rotate(edge, angle, axes=rotate_axes[k], reshape=False, order=0, mode='constant', cval=0.0)
    return image, label


# z, y, x     0, 1, 2
def rot_from_y_x(image, label):
    # k = np.random.randint(0, 4)
    image = np.rot90(image, 2, axes=(1, 2))  # rot along z axis
    label = np.rot90(label, 2, axes=(1, 2))

    return image, label


def flip_xz_yz(image, label):
    flip_id = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
    image = np.ascontiguousarray(image[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    label = np.ascontiguousarray(label[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    return image, label


def intensity_shift(image):
    shift_value = random.uniform(-0.1, 0.1)

    image = image + shift_value
    return image


def intensity_scale(image):
    scale_value = random.uniform(0.9, 1.1)

    image = image * scale_value
    return image


# x: label   n: num of classes
# to one hot vector
def make_one_hot_3d(x, n):
    one_hot = torch.zeros([n, x.shape[0], x.shape[1], x.shape[2]])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for v in range(x.shape[2]):
                one_hot[np.int(x[i, j, v]), i, j, v] = 1
    return one_hot


class RandomGenerator(object):
    def __init__(self, output_size, mode):
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        min_value = np.min(image)
        # centercop
        # crop alongside with the ground truth

        index = np.nonzero(label)
        index = np.transpose(index)

        z_min = np.min(index[:, 0])
        z_max = np.max(index[:, 0])
        y_min = np.min(index[:, 1])
        y_max = np.max(index[:, 1])
        x_min = np.min(index[:, 2])
        x_max = np.max(index[:, 2])

        # patch_y = np.int(self.output_size[1] / 6)
        # patch_x = np.int(self.output_size[2] / 4)

        # middle point
        z_middle = np.int((z_min + z_max) / 2)
        y_middle = np.int((y_min + y_max) / 2)
        x_middle = np.int((x_min + x_max) / 2)

        if random.random() > 0.3:
            Delta_z = np.int((z_max - z_min) / 3)  # 3
            Delta_y = np.int((y_max - y_min) / 8)  # 8
            Delta_x = np.int((x_max - x_min) / 8)  # 8

        else:
            Delta_z = np.int((z_max - z_min) / 2) + self.output_size[0]
            Delta_y = np.int((y_max - y_min) / 8)
            Delta_x = np.int((x_max - x_min) / 8)

        # random number of x, y, z
        z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)
        y_random = random.randint(y_middle - Delta_y, y_middle + Delta_y)
        x_random = random.randint(x_middle - Delta_x, x_middle + Delta_x)

        # crop patch
        crop_z_down = z_random - np.int(self.output_size[0] / 2)
        crop_z_up = z_random + np.int(self.output_size[0] / 2)
        crop_y_down = y_random - np.int(self.output_size[1] / 2)
        crop_y_up = y_random + np.int(self.output_size[1] / 2)
        crop_x_down = x_random - np.int(self.output_size[2] / 2)
        crop_x_up = x_random + np.int(self.output_size[2] / 2)


        if crop_z_down < 0 or crop_z_up > image.shape[0]:
            delta_z = np.maximum(np.abs(crop_z_down), np.abs(crop_z_up - image.shape[0]))
            image = np.pad(image, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=min_value)
            label = np.pad(label, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=0.0)

            crop_z_down = crop_z_down + delta_z
            crop_z_up = crop_z_up + delta_z

        if crop_y_down < 0 or crop_y_up > image.shape[1]:
            delta_y = np.maximum(np.abs(crop_y_down), np.abs(crop_y_up - image.shape[1]))
            image = np.pad(image, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=min_value)
            label = np.pad(label, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=0.0)

            crop_y_down = crop_y_down + delta_y
            crop_y_up = crop_y_up + delta_y

        if crop_x_down < 0 or crop_x_up > image.shape[2]:
            delta_x = np.maximum(np.abs(crop_x_down), np.abs(crop_x_up - image.shape[2]))
            image = np.pad(image, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=min_value)
            label = np.pad(label, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=0.0)

            crop_x_down = crop_x_down + delta_x
            crop_x_up = crop_x_up + delta_x

        label = label[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
        image = image[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
        # print(image.shape)

        # cause 0.9998
        label = np.round(label)

        # data augmentation
        if self.mode == 'train':
            # data augmentation
            # if random.random() > 0.5:
            # image, label = random_rot_flip(image, label)
            # image, label = rot_from_y_x(image, label)
            # if random.random() > 0.5:
            # image, label = flip_xz_yz(image, label)
            if random.random() > 0.5:  # elif random.random() > 0.5:
                image = intensity_shift(image)
            if random.random() > 0.5:  # elif random.random() > 0.5:
                image = intensity_scale(image)
            if random.random() > 0.5:  # elif random.random() > 0.5:
                image, label = random_rotate(image, label, min_value)
                label = np.round(label)

        image = torch.from_numpy(image.astype(np.float)).unsqueeze(0).float()
        label = torch.from_numpy(label.astype(np.float32)).float()

        # image = torch.from_numpy(image.astype(np.float)).unsqueeze(0)
        # label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label.long()}
        return sample


class verse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, num_classes, args):
        self.h = args.img_size[0] / 2
        self.w = args.img_size[1] / 2
        self.l = args.img_size[2] / 2
        self.num_coefficient = args.coefficient
        self.split = split
        name = []
        center = []
        contour = []
        if self.split != 'test_vol':
            with open(os.path.join(list_dir, self.split + '.csv'), 'r') as file:
                reader = csv.reader(file)

                for row in reader:
                    name.append(row[0])

                    center_value = row[1][1: -1]
                    # print(type(center_value))   # 'str'
                    center_value = np.array([np.float(x) for x in center_value.split(', ')])
                    center.append(center_value)

                    contour_value = row[2][1: -1]
                    # print(type(coefficient_value))
                    contour_value = np.array([np.float(x) for x in contour_value.split(', ')])
                    contour_value = contour_value.reshape(4000, 3)
                    contour.append(contour_value)
        self.name = name
        self.center = center
        self.contour_index = contour

        # self.mean_value_coefficient = np.load(list_dir + '/mean_value.npy')
        # print('coefficient shape: ', self.mean_value_coefficient.shape)
        self.data_dir = base_dir
        self.num_classes = num_classes

        '''
        
        contour_index = []
        with open('/mnt/data/youxin/python_files/eigencontour/output1.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                contour_value = row
                # print(contour_value)
                # contour_value = np.array([np.float(x) for x in contour_value.split(', ')])         'list'
                contour_value = np.array([np.float(x) for x in contour_value])
                contour_value = contour_value.reshape(10000, 3)

                contour_index.append(contour_value)
        self.contour_index = contour_index
        '''
        if self.split == 'test_vol':
            self.name = open(os.path.join(list_dir, self.split + '.txt')).readlines()

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.name[idx].strip('\n')
            # print(slice_name)
            img_path = os.path.join(self.data_dir + '/img', slice_name)
            image = sitk.ReadImage(img_path)
            label_path = os.path.join(self.data_dir + '/label', slice_name)
            label = sitk.ReadImage(label_path)
            # edge_path = os.path.join(self.data_dir + '/edge', slice_name)
            # edge = sitk.ReadImage(edge_path)
            origin = np.array(image.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(image.GetSpacing())
            image = sitk.GetArrayFromImage(image)
            label = sitk.GetArrayFromImage(label)
            # edge = sitk.GetArrayFromImage(edge)

            # normalize
            center = self.center[idx]
            center = center / [self.h, self.w, self.l]

            # normalize for coefficients
            # coefficient = self.coefficient[idx][0: self.num_coefficient] / self.mean_value_coefficient[0: self.num_coefficient]
            contour_index = self.contour_index[idx]

        elif self.split == "val":
            slice_name = self.name[idx].strip('\n')
            img_path = os.path.join(self.data_dir + '/img', slice_name)
            image = sitk.ReadImage(img_path)
            label_path = os.path.join(self.data_dir + '/label', slice_name)
            label = sitk.ReadImage(label_path)
            # edge_path = os.path.join(self.data_dir + '/edge', slice_name)
            # edge = sitk.ReadImage(edge_path)
            origin = np.array(image.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(image.GetSpacing())
            image = sitk.GetArrayFromImage(image)
            label = sitk.GetArrayFromImage(label)
            # edge = sitk.GetArrayFromImage(edge)
            # normalize
            center = self.center[idx]
            center = center / [self.h, self.w, self.l]

            # normalize for coefficients
            # coefficient = self.coefficient[idx][0: self.num_coefficient] / self.mean_value_coefficient[0: self.num_coefficient]
            contour_index = self.contour_index[idx]

        else:
            slice_name = self.name[idx].strip('\n')
            img_path = os.path.join(self.data_dir, slice_name + '_img.nii.gz')  # normalized_image
            image = sitk.ReadImage(img_path)
            label_path = os.path.join(self.data_dir, slice_name + '_gt.nii.gz')
            label = sitk.ReadImage(label_path)
            pred_path = os.path.join(self.data_dir, slice_name + '_pred.nii.gz')
            pred = sitk.ReadImage(pred_path)

            origin = np.array(image.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(image.GetSpacing())
            image = sitk.GetArrayFromImage(image)
            label = sitk.GetArrayFromImage(label)
            pred = sitk.GetArrayFromImage(pred)
            '''
            # normalize
            center = self.center[idx]
            center = center / [self.h, self.w, self.l]
            # normalize for coefficients
            coefficient = self.coefficient[idx][0: self.num_coefficient] / self.mean_value_coefficient[0: self.num_coefficient] 
            '''

        if self.split != "test_vol":

            label[label < 0.5] = 0.0  # maybe some voxels is a minus value
            label[label > 0.5] = 1.0

            # edge[edge < 0.5] = 0.0  # maybe some voxels is a minus value
            # edge[edge > 0.5] = 1.0
            min_value = np.min(image)

            # data augmentation
            if self.split == 'train':

                # if random.random() > 0.5:
                    # image, label = flip_xz_yz(image, label)
                if random.random() > 0.5:  # elif random.random() > 0.5:
                    image = intensity_shift(image)
                if random.random() > 0.5:  # elif random.random() > 0.5:
                    image = intensity_scale(image)
                # if random.random() > 0.5:  # elif random.random() > 0.5:
                    # image, label = random_rotate(image, label, min_value)
                    # label = np.round(label)
                    # edge = np.round(edge)
            image = torch.from_numpy(image.astype(np.float)).unsqueeze(0).float()
            label = torch.from_numpy(label.astype(np.float32)).float()
            # edge = torch.from_numpy(edge.astype(np.float32)).float()
            center = torch.from_numpy(center.astype(np.float)).float()
            # coefficient = torch.from_numpy(coefficient.astype(np.float)).float()
            contour_index = torch.from_numpy(contour_index.astype(np.float)).float()
            sample = {'image': image, 'label': label, 'center': center, 'contour': contour_index}
            sample['case_name'] = self.name[idx].strip('\n')

            sample['origin'] = origin
            sample['spacing'] = spacing

        if self.split == "test_vol":
            image = torch.from_numpy(image.astype(np.float)).float()
            label = torch.from_numpy(label.astype(np.float32)).float()
            pred = torch.from_numpy(pred.astype(np.float32)).float()
            sample = {'image': image, 'label': label, 'pred': pred}
            sample['case_name'] = self.name[idx].strip('\n')

            sample['origin'] = origin
            sample['spacing'] = spacing
        return sample



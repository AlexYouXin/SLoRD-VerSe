import cv2
from sklearn.mixture import GaussianMixture
import os
import numpy as np
import SimpleITK as sitk
from skimage.segmentation import find_boundaries
import random
from medpy import metric


def min_max(data, min_, max_):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = data * (max_ - min_) + min_
    return data


def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
       for filename in files:
            Filelist.append(filename)
    return Filelist



def calculate_metric_percase(pred, gt, space):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=space)  # , voxelspacing=space
        hd = metric.binary.hd(pred, gt, voxelspacing=space)
        # hd = metric.binary.hd(pred, gt, voxelspacing=space)
        return dice, hd95, hd

    else:
        return 0, 30, 20


def evaluation_metric(pred, gt, space):
    index = np.nonzero(gt)
    index = np.transpose(index)

    flatten_label = gt.flatten()
    list_label = flatten_label.tolist()
    set_label = set(list_label)
    print('different values:', set_label)
    length = len(set_label)
    # print('number of different values: ', length)
    list_label_ = list(set_label)
    list_label_ = np.array(list_label_).astype(np.int)

    index = np.zeros(classes)
    metric_list = np.zeros((classes, 3))
    for i in range(1, length):
        metric_list[list_label_[i], :] = calculate_metric_percase(pred == list_label_[i], gt == list_label_[i], space)
        index[list_label_[i]] += 1

    return metric_list, index

if __name__ == "__main__":
    # load image
    out_path = 'xxxxxx/'

    classes = 26

    sample_list = open(os.path.join('xxxx/evaluation', 'test.txt')).readlines()

    num_file = len(sample_list)

    metric_list = 0.0
    index = 0.0
    ave_dice = 0
    ave_hd95 = 0
    ave_hd = 0
    dice_case = []
    hd95_case = []
    hd_case = []

    for i in range(num_file):
        file = sample_list[i].strip('\n')
        pred = sitk.ReadImage(out_path + file.replace('.gz', '.gz_fusion.nii.gz'))
        label = sitk.ReadImage(out_path + file.replace('.gz', '.gz_gt.nii.gz'))
        origin = pred.GetOrigin()
        space = pred.GetSpacing()
        direction = pred.GetDirection()
        # print('origin:', origin)
        pred = sitk.GetArrayFromImage(pred)
        label = sitk.GetArrayFromImage(label)

        metric_i, index_i = evaluation_metric(pred, label, space)


        num = np.count_nonzero(index_i)
        mean_dice = np.sum(metric_i, axis=0)[0] / num
        mean_hd95 = np.sum(metric_i, axis=0)[1] / num
        mean_hd = np.sum(metric_i, axis=0)[2] / num
        print('case: %s, mean_dice: %f, mean_hd95: %f, mean_hd: %f' % (file, mean_dice, mean_hd95, mean_hd))
        print('space: ', space)
        ave_dice = ave_dice + mean_dice
        ave_hd95 = ave_hd95 + mean_hd95
        ave_hd = ave_hd + mean_hd
        dice_case.append(mean_dice)
        hd95_case.append(mean_hd95)
        hd_case.append(mean_hd)
        index += index_i
        metric_list += metric_i


    for i in range(1, classes):
        metric_list[i, :] = metric_list[i, :] / index[i]
        print('Mean class: %d, mean_dice %f, mean_hd95 %f, mean_hd: %f' % (i, metric_list[i][0], metric_list[i][1], metric_list[i][2]))

    performance = np.sum(metric_list, axis=0)[0] / (classes - 1)
    mean_hd95 = np.sum(metric_list, axis=0)[1] / (classes - 1)
    mean_hd = np.sum(metric_list, axis=0)[2] / (classes - 1)
    print('Testing performance on classes: mean_dice : %f mean_hd95 : %f mean_hd: %f' % (performance, mean_hd95, mean_hd))

    c_dice = np.sum(metric_list[1:8, :], axis=0)[0] / 7
    c_hd95 = np.sum(metric_list[1:8, :], axis=0)[1] / 7
    c_hd = np.sum(metric_list[1:8, :], axis=0)[2] / 7
    t_dice = np.sum(metric_list[8:20, :], axis=0)[0] / 12
    t_hd95 = np.sum(metric_list[8:20, :], axis=0)[1] / 12
    t_hd = np.sum(metric_list[8:20, :], axis=0)[2] / 12
    l_dice = np.sum(metric_list[20:26, :], axis=0)[0] / 6
    l_hd95 = np.sum(metric_list[20:26, :], axis=0)[1] / 6
    l_hd = np.sum(metric_list[20:26, :], axis=0)[2] / 6

    print('Testing performance on classes: c_dice : %f c_hd95 : %f c_hd : %f' % (c_dice, c_hd95, c_hd))
    print('Testing performance on classes: t_dice : %f t_hd95 : %f t_hd : %f' % (t_dice, t_hd95, t_hd))
    print('Testing performance on classes: l_dice : %f l_hd95 : %f l_hd : %f' % (l_dice, l_hd95, l_hd))

    ave_dice = ave_dice / 40
    ave_hd95 = ave_hd95 / 40
    ave_hd = ave_hd / 40
    print('Testing performance on cases: mean_dice : %f mean_hd95 : %f  mean_hd : %f' % (ave_dice, ave_hd95, ave_hd))
    # medium
    # sort
    dice_case = np.sort(dice_case)
    hd95_case = np.sort(hd95_case)
    hd_case = np.sort(hd_case)
    middle_dice = (dice_case[19] + dice_case[20]) / 2
    middle_hd95 = (hd95_case[19] + hd95_case[20]) / 2
    middle_hd = (hd_case[19] + hd_case[20]) / 2
    print('Testing performance on cases: middle_dice : %f middle_hd95 : %f middle_hd : %f' % (middle_dice, middle_hd95, mean_hd))


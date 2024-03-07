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

    contour_combine[:, :, 2] = out[:, 0].flatten(1).clamp(0, x)
    # contour_combine[:, :, 2] = contour_combine[:, :, 2]
    contour_combine[:, :, 1] = out[:, 1].flatten(1).clamp(0, y)
    contour_combine[:, :, 0] = out[:, 2].flatten(1).clamp(0, z)

    return contour_combine


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    # print('Total number of parameters: %d' % num_params)
    return num_params, net


class DiceLoss(nn.Module):
    def __init__(self, n_classes, weight):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        # print(intersect.item())
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        # print(loss.item())
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        # print('outputs, targets after one-hot:', inputs.shape, target.shape)           # ([1, 2, 64, 64, 64])  ([1, 2, 64, 64, 64])
        # allocate weight for segmentation of each class
        # if weight is None:
        # weight = [1] * self.n_classes
        weight = self.weight

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        # print('num classes:', self.n_classes)         # 2
        for i in range(0, self.n_classes):
            # print(inputs.shape, target.shape)              # torch.Size([1, 26, 128, 128, 128]) torch.Size([1, 26, 128, 128, 128])
            # print('min and max: ', torch.min(inputs[:, i, :, :, :]), torch.min(target[:, i, :, :, :]), torch.max(inputs[:, i, :, :, :]), torch.max(target[:, i, :, :, :]))
            dice_loss = self._dice_loss(inputs[:, i], target[:, i])
            # print(dice_loss.item())
            class_wise_dice.append(1.0 - dice_loss.item())
            loss += dice_loss * weight[i]
        # return loss / self.n_classes
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95

    else:
        return 0, 30


def threshold_operation(pred, label_index, thre):
    length = len(label_index)
    refine_prediction = np.zeros_like(pred)
    for index in range(0, length - 1):
        pred_index = pred[pred == label_index[index]].astype(np.int)
        num_voxel = np.count_nonzero(pred_index)
        if label_index[index] <= 7 and label_index[index] >= 1:
            threshold_value = thre[0]
        elif label_index[index] <= 19 and label_index[index] >= 8:
            threshold_value = thre[1]
        elif label_index[index] <= 25 and label_index[index] >= 20:
            threshold_value = thre[2]
        if num_voxel > threshold_value:
            refine_prediction[pred == label_index[index]] = label_index[index]
    refine_prediction[pred == label_index[length - 1]] = label_index[length - 1]

    return refine_prediction


def center_calculation(volume):
    index = np.nonzero(volume)
    index = np.transpose(index)

    z_mean = np.int(np.mean(index[:, 0]))
    y_mean = np.int(np.mean(index[:, 1]))
    x_mean = np.int(np.mean(index[:, 2]))
    center_volume = [z_mean, y_mean, x_mean]

    return center_volume


def convex_hull(prediction_):
    prediction_ = np.ascontiguousarray(prediction_)
    # print('mask shape: ', prediction_.shape)
    mask = convex_hull_image(prediction_)

    return mask


def test_single_volume(image_, pred_, label_, net0, net1, net2, classes, patch_size, test_save_path=None, case=None, origin=None,
                       spacing=None):
    image_, label_, pred_ = image_.squeeze(0).cpu().detach().numpy(), label_.squeeze(
        0).cpu().detach().numpy(), pred_.squeeze(0).cpu().detach().numpy()
    print('previous image shape: ', image_.shape[0], image_.shape[1], image_.shape[2])
    label_[label_ < 0.5] = 0.0  # maybe some voxels is a minus value
    label_[label_ > 25.5] = 0.0

    label_ = np.round(label_)

    threshold = [4686, 8204, 14269]

    min_value = np.min(image_)
    # image padding
    padding_size = [np.int(patch_size[0] / 2), np.int(patch_size[1] / 2), np.int(patch_size[2] / 2)]
    '''
    # get non-zeros index
    index = np.nonzero(pred_)
    index = np.transpose(index)
    z_min = np.min(index[:, 0])
    z_max = np.max(index[:, 0])
    y_min = np.min(index[:, 1])
    y_max = np.max(index[:, 1])
    x_min = np.min(index[:, 2])
    x_max = np.max(index[:, 2])
    '''
    image_ = np.pad(image_, (
    (padding_size[0], padding_size[0]), (padding_size[1], padding_size[1]), (padding_size[2], padding_size[2])),
                    'constant', constant_values=min_value)
    label_ = np.pad(label_, (
    (padding_size[0], padding_size[0]), (padding_size[1], padding_size[1]), (padding_size[2], padding_size[2])),
                    'constant', constant_values=0)
    pred_ = np.pad(pred_, (
    (padding_size[0], padding_size[0]), (padding_size[1], padding_size[1]), (padding_size[2], padding_size[2])),
                   'constant', constant_values=0)
    z, y, x = pred_.shape

    z_size = np.int(patch_size[0] / 2)
    y_size = np.int(patch_size[1] / 2)
    x_size = np.int(patch_size[2] / 2)

    # before labels
    before_labels = sorted(list(set(pred_[pred_ != 0])))
    print('before: ', before_labels, len(before_labels))

    # threshold selection
    pred_ = pred_.astype(np.int)
    refined_prediction0 = threshold_operation(pred_.copy(), before_labels, threshold)
    after_label = sorted(list(set(refined_prediction0[refined_prediction0 != 0])))
    print('after: ', after_label, len(after_label))

    net0.eval()
    print('refined_prediction0 shape: ', refined_prediction0.shape)
    refined_prediction1 = np.zeros_like(refined_prediction0)

    for label_index in after_label:
        prediction = np.zeros_like(refined_prediction0)
        prediction[refined_prediction0 == label_index] = 1

        # print('prediction shape: ', prediction.shape)
        # hull = convex_hull(prediction.copy())
        hull = prediction
        # calculate the center of hull_index
        '''
        index = np.nonzero(hull)
        index = np.transpose(index)
        z_min = np.min(index[:, 0])
        z_max = np.max(index[:, 0])
        y_min = np.min(index[:, 1])
        y_max = np.max(index[:, 1])
        x_min = np.min(index[:, 2])
        x_max = np.max(index[:, 2])
        hull_crop = hull[z_min: z_max, y_min: y_max, x_min: x_max]
        '''
        center = center_calculation(hull.copy())
        # print('center: ', center, label_index)
        # patch crop centering at center
        patch_center = image_[center[0] - z_size: center[0] + z_size, center[1] - y_size: center[1] + y_size,
                       center[2] - x_size: center[2] + x_size].copy()
        # network input
        input = torch.from_numpy(patch_center).unsqueeze(0).unsqueeze(
            0).float().cuda()

        outputs0, center0, coefficient0 = net0(input)
        outputs1, center1, coefficient1 = net1(input)
        outputs2, center2, coefficient2 = net2(input)

        outputs0 = torch.softmax(outputs0, dim=1).squeeze(0)
        outputs1 = torch.softmax(outputs1, dim=1).squeeze(0)
        outputs2 = torch.softmax(outputs2, dim=1).squeeze(0)

        outputs0 = outputs0.cpu().detach().numpy()
        outputs1 = outputs1.cpu().detach().numpy()
        outputs2 = outputs2.cpu().detach().numpy()

        outputs = (outputs0 + outputs1 + outputs2) / 3

        out = np.argmax(outputs, axis=0)

        out = out * label_index

        t = refined_prediction1[center[0] - z_size: center[0] + z_size, center[1] - y_size: center[1] + y_size,
        center[2] - x_size: center[2] + x_size].copy()
        t[out == label_index] = label_index
        refined_prediction1[center[0] - z_size: center[0] + z_size, center[1] - y_size: center[1] + y_size,
        center[2] - x_size: center[2] + x_size] = t.copy()



    image_ = image_[padding_size[0]: z - padding_size[0], padding_size[1]: y - padding_size[1],
             padding_size[2]: x - padding_size[2]]
    label_ = label_[padding_size[0]: z - padding_size[0], padding_size[1]: y - padding_size[1],
             padding_size[2]: x - padding_size[2]]
    pred_ = pred_[padding_size[0]: z - padding_size[0], padding_size[1]: y - padding_size[1],
            padding_size[2]: x - padding_size[2]]
    refined_prediction0 = refined_prediction0[padding_size[0]: z - padding_size[0],
                          padding_size[1]: y - padding_size[1], padding_size[2]: x - padding_size[2]]
    refined_prediction1 = refined_prediction1[padding_size[0]: z - padding_size[0],
                          padding_size[1]: y - padding_size[1], padding_size[2]: x - padding_size[2]]

    index = np.nonzero(label_)
    index = np.transpose(index)
    z_min = np.min(index[:, 0])
    z_max = np.max(index[:, 0])
    y_min = np.min(index[:, 1])
    y_max = np.max(index[:, 1])
    x_min = np.min(index[:, 2])
    x_max = np.max(index[:, 2])

    flatten_label = label_.flatten()
    list_label = flatten_label.tolist()
    set_label = set(list_label)
    print('different values:', set_label)
    length = len(set_label)
    # print('number of different values: ', length)
    list_label_ = list(set_label)
    list_label_ = np.array(list_label_).astype(np.int)

    index = np.zeros(classes)
    metric_list = np.zeros((classes, 2))
    for i in range(1, length):
        metric_list[list_label_[i], :] = calculate_metric_percase(
            refined_prediction1[z_min: z_max, y_min: y_max, x_min: x_max] == list_label_[i],
            label_[z_min: z_max, y_min: y_max, x_min: x_max] == list_label_[i])
        index[list_label_[i]] += 1

    binary_map = refined_prediction1[z_min: z_max, y_min: y_max, x_min: x_max].copy()
    binary_map[binary_map >= 1] = 1
    binary_map[binary_map < 1] = 0

    binary_label = label_[z_min: z_max, y_min: y_max, x_min: x_max].copy()
    binary_label[binary_label >= 1] = 1
    binary_label[binary_label < 1] = 0

    binary_metric = calculate_metric_percase(binary_map, binary_label)

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image_.astype(np.float32))
        thre_itk = sitk.GetImageFromArray(refined_prediction0.astype(np.float32))
        refine_itk = sitk.GetImageFromArray(refined_prediction1.astype(np.float32))
        coarse_itk = sitk.GetImageFromArray(pred_.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label_.astype(np.float32))

        origin = origin.flatten()
        spacing = spacing.flatten()
        origin = origin.numpy()
        spacing = spacing.numpy()

        print('origin and spacing: ', origin, spacing)

        img_itk.SetOrigin(origin)
        img_itk.SetSpacing(spacing)
        coarse_itk.SetOrigin(origin)
        coarse_itk.SetSpacing(spacing)
        refine_itk.SetOrigin(origin)
        refine_itk.SetSpacing(spacing)
        lab_itk.SetOrigin(origin)
        lab_itk.SetSpacing(spacing)
        thre_itk.SetOrigin(origin)
        thre_itk.SetSpacing(spacing)

        sitk.WriteImage(coarse_itk, test_save_path + '/' + case + "_coarse.nii.gz")
        sitk.WriteImage(refine_itk, test_save_path + '/' + case + "_refine.nii.gz")
        sitk.WriteImage(thre_itk, test_save_path + '/' + case + "_thre.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list, index, binary_metric


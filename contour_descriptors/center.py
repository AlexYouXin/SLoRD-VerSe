import cv2 as cv
import SimpleITK as sitk
from scipy.spatial import distance
from utils import *
import numpy as np
from mayavi import mlab


class TOsmallError(Exception):
    pass


def boundingRect3D(instance_mask):
    """Calculate the coordinates and width, height and depth of the bounding box"""
    # Get the index of a non-zero pixel
    nonzero_pixels = instance_mask.nonzero()

    # Calculate the coordinates and dimensions of the bounding box
    x_min = min(nonzero_pixels[2])
    y_min = min(nonzero_pixels[1])
    z_min = min(nonzero_pixels[0])
    x_max = max(nonzero_pixels[2])
    y_max = max(nonzero_pixels[1])
    z_max = max(nonzero_pixels[0])

    # Calculate width, height and depth
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    d = z_max - z_min + 1

    return x_min, y_min, z_min, w, h, d


def inner_dot(instance_mask, point):
    """Checks whether the given point is inside the instance mask (binary mask)"""
    xp, yp, zp = point
    d, h, w = instance_mask.shape
    bool_inst_mask = instance_mask.astype(bool)
    # Create a matrix of zeros of the same size as the instance mask, representing the location of the points
    neg_bool_inst_mask = 1 - bool_inst_mask
    dot_mask = np.zeros(instance_mask.shape)
    instd, insth, instw = instance_mask.shape
    # Mark the position of the point as 1
    dot_mask[zp][yp][xp] = 1
    # Check if the position of the point is within the image boundary
    if yp + 2 >= h or yp - 2 < 0 or xp + 2 >= w or xp - 2 < 0 or zp + 2 >= d or zp - 2 < 0:
        return False
    # Create a 3x3 padding matrix for expansion point locations
    fill_mask = np.zeros((5, 5, 5))
    fill_mask.fill(1)
    # Extension point location
    dot_mask[zp - 2:zp + 3, yp - 2:yp + 3, xp - 2:xp + 3] = fill_mask
    # Use logical operations to check if the point is inside the instance mask
    not_inner = (neg_bool_inst_mask * dot_mask).any()
    # print(np.sum(neg_bool_inst_mask),np.sum(dot_mask))
    # print('neg_bool',np.unique(dot_mask))
    return not not_inner


def centerdot3D(instance_mask):
    """Find the centroid of a 3D instance mask via Euclidean distance"""
    # boundingorder x, y
    bool_inst_mask = instance_mask.astype(bool)
    # Compute bounding box information for instance masks
    x, y, z, w, h, d = boundingRect3D(instance_mask)
    # Calculate the center coordinates of the bounding box (float)
    avg_center_float = (x + w / 2, y + h / 2, z + d / 2)  # w,h,d
    # Convert center coordinates to integer type
    avg_center = (int(avg_center_float[0]), int(avg_center_float[1]), int(avg_center_float[2]))
    # Create a temporary zero matrix that checks if the center point is in the instance mask
    temp = np.zeros(instance_mask.shape)
    temp[int(avg_center[2])][int(avg_center[1])][int(avg_center[0])] = 1
    if (bool_inst_mask == temp).any() and inner_dot(instance_mask, avg_center):
        return avg_center_float
    else:
        # Get the coordinates of non-zero points in the instance mask
        inst_mask_d, inst_mask_h, inst_mask_w = np.where(instance_mask)

        # get gradient_map
        gradient_map = get_gradient(instance_mask)
        grad_d, grad_h, grad_w = np.where(gradient_map == 1)

        # Store non-zero point coordinates and point coordinates on the gradient map as arrays respectively
        # inst_points
        inst_points = np.array([[inst_mask_w[i], inst_mask_h[i], inst_mask_d[i]] for i in range(len(inst_mask_h))])
        # edge_points
        bounding_order = np.array([[grad_w[i], grad_h[i], grad_d[i]] for i in range(len(grad_h))])

        center_distance = inst_mask_w - int(avg_center_float[0])

        # Calculate the Euclidean distance of each non-zero point to a point on the gradient map
        distance_result = distance.cdist(inst_points, bounding_order, 'euclidean')
        sum_distance = np.sum(distance_result, 1)

        # Find the non-zero point with the minimum total distance as the center point
        center_index = np.argmin(sum_distance)
        center_dot = (inst_points[center_index][0], inst_points[center_index][1], inst_points[center_index][2])
        times_num = 0

        # Check whether the center point meets the inner_dot condition, and make corrections if it does not.
        while not inner_dot(instance_mask, center_dot):
            times_num += 1
            sum_distance = np.delete(sum_distance, center_index)
            if len(sum_distance) == 0:
                print('no center')
                # raise TOsmallError
                return (0, 0, 0)

            center_index = np.argmin(sum_distance)
            center_dot = (inst_points[center_index][0], inst_points[center_index][1], inst_points[center_index][2])
        return center_dot


if __name__ == "__main__":
    # file path
    img_folder = 'D:/xxxxxxxx/'
    img_file_name = "xxxxxxxxx.nii.gz"

    # Get file list
    img_file_path = os.path.join(img_folder, img_file_name)

    img = sitk.ReadImage(img_file_path)

    img_array = sitk.GetArrayFromImage(img)
    unique_labels = sorted(list(set(img_array[img_array != 0])))

    for label in unique_labels:
        # Find the indices of all points equal to the current label value
        img_mask = (img_array == label).astype(np.uint8)

        center = centerdot3D(img_mask)
        print(center)
        gradient_volume = get_gradient(img_mask)

        # Create a 3D scene
        mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))

        # Visualizing 3D arrays
        # src = mlab.pipeline.scalar_field(spine_volume)
        src = mlab.pipeline.scalar_field(gradient_volume)
        mlab.pipeline.volume(src,  color=(0.8, 0.8, 0.8))  # Use green colormap

        mlab.points3d(center[2], center[1], center[0], color=(1, 0, 0), scale_factor=1.5)

        # Set interactive mode
        mlab.view(azimuth=0, elevation=90, distance=300)
        mlab.show()



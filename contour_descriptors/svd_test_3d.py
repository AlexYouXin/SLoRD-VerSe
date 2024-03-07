import torch
import math
from center_to_contour_distance import *
from center import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import vtkmodules.all as vtk
import vtkmodules.util.numpy_support as vtk_np
from mayavi import mlab
import numpy as np


def singular_value(S):
    """Calculates the number of singular values to retain and the corresponding ratio of their sum"""
    num = len(S)
    k = 800
    ratio = np.sum(S[0: k]) / np.sum(S)
    return k, ratio


if __name__ == "__main__":
    img_folder = 'D:/xxxxxxxx/'
    img_file_name = "xxxxxxxx.nii.gz"

    # Calculate the maximum distance from the image center to the image edge
    width = 40
    height = 30
    z = 30
    max_dist = math.sqrt(width / 2 * width / 2 + height / 2 * height / 2 + z / 2 * z / 2)

    # Normalized distance data
    mat = np.load('verse19_distance_final_5.npy')
    mat = mat / max_dist
    # mat = mat.flatten(1)
    print('mat shape: ', mat.shape)
    n, r, c = mat.shape

    mat = np.reshape(mat, (n, -1))
    # Computes the singular value decomposition (SVD) of the distance matrix
    n, l = mat.shape

    k = np.minimum(n, l)
    U, S, V = np.linalg.svd(mat)
    # prints the shapes of U, S, and V along with the singular values.
    print('U shape: ', U.shape)
    print('S shape: ', S.shape)
    print('V shape: ', V.shape)
    # U shape:  (1725, 1725)
    # S shape:  (1725,)
    # V shape:  (2664, 2664)
    # print('S: ', S)

    # singular value
    # Calculate the cardinality and singular value ratio of singular values
    num_base, ratio = singular_value(S)
    print('ratio of singular value: ', ratio)
    print('num base of singular value: ', num_base)

    verse = 5
    dim = (int(360 / verse), int(180 / verse + 1))

    img = sitk.ReadImage(img_folder + img_file_name)

    img_array = sitk.GetArrayFromImage(img)

    unique_labels = sorted(list(set(img_array[img_array != 0])))
    print('uniqur label: ', unique_labels)
    # Need to binarize the values of the mask
    label = unique_labels[6]
    img_mask = (img_array == label).astype(np.uint8)

    center = centerdot3D(img_mask)
    gradient_volume = get_gradient(img_mask)
    print('shape: ', gradient_volume.shape)  # shape:  (159, 160, 115)

    points, center_x, center_y, center_z = getOrientedPoints(img_mask, verse, dim)

    print('center: ', center_z, center_y, center_x)
    print('size of points: ', len(points), len(points[0]))  # size of points:  72 37

    x_len, y_len, z_len = bbox(img_mask)

    data = dict()
    data['bbox'] = (center_x, center_y, center_z, x_len, y_len, z_len)
    data['r'] = points
    data['center'] = (center_x, center_y, center_z)

    out, xr, yr, zr, r1 = decoding_theta(data, dim)

    # Create a 3D scene
    mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))

    # Visualizing 3D arrays

    src = mlab.pipeline.scalar_field(gradient_volume)
    mlab.pipeline.volume(src, color=(0.8, 0.8, 0.8))  # Renders the volume with a gray color

    mlab.points3d(center[2], center[1], center[0], color=(1, 0, 0), scale_factor=1.5)

    points_array = np.array(out)
    mlab.points3d(points_array[:, 2], points_array[:, 1], points_array[:, 0], color=(0, 0.54, 0.54), scale_factor=0.7)

    # Generate mesh
    # mesh = mlab.mesh(zr, yr, xr, scalars=r1)

    # Set interactive mode
    mlab.view(azimuth=0, elevation=90, distance=100)
    mlab.show()

    # Get the basis vectors corresponding to the first num_base singular values and generate contour points
    # Take out k bases
    contour_points = np.zeros((num_base, dim[0] * dim[1], 3))
    base_vector_array = V[0: num_base, :]
    for i in range(num_base):
        base_vector = base_vector_array[i]
        # multiply by max_dist
        result = base_vector * max_dist * 10

        data['r'] = list(result.reshape(dim[0], dim[1]))

        out, _, _, _, _ = decoding_theta(data, dim)
        # print(out['contour_pts'].shape, type(out['contour_pts']))
        points_array = np.array(out)
        contour_points[i, :, 0] = points_array[:, 0].flatten()
        contour_points[i, :, 1] = points_array[:, 1].flatten()
        contour_points[i, :, 2] = points_array[:, 2].flatten()

    # Visualization of substrate
    contour_points = contour_points.astype(np.int32)
    print('base contour vector shape', contour_points.shape)

    # contour_points = contour_points[:, 0: 100, :]

    my_list = [1, 2, 3, 4, 5, 6, 7, 10, 12, 15, 20, 25, 30, 40, 50, 80, 100, 150, 250, 300, 500]

    for plt_index in range(1, num_base + 1):
        if plt_index in my_list:
            mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))

            # Visualizing 3D arrays

            src = mlab.pipeline.scalar_field(gradient_volume)
            mlab.pipeline.volume(src, color=(0.8, 0.8, 0.8))  # Use gray colormap
            mlab.points3d(center[2], center[1], center[0], color=(1, 0, 0), scale_factor=1.5)
            mlab.points3d(contour_points[plt_index - 1, :, 2], contour_points[plt_index - 1, :, 1],
                          contour_points[plt_index - 1, :, 0], color=(0, 0.54, 0.54), scale_factor=0.7)

            mlab.view(azimuth=0, elevation=90, distance=100)
            mlab.show()



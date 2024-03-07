import math
import random
import cv2
from utils import *
from center import *
# from center_compute import *
import numpy as np
import vtkmodules.all as vtk
import vtkmodules.util.numpy_support as vtk_np
from mayavi import mlab

verse = 5


# input instance with only one contour
def getOrientedPoints(instance, verse, dim, mode='ese_ori'):         # mode=ese_ori
    """
    Get the orientation point and center point in the spherical coordinate system according to the provided parameters
    """
    # first get center point
    instance = instance.astype(np.uint8)         # np.int
    if mode == 'ese_ori':
        center_x, center_y, center_z = centerdot3D(
            instance)
    elif mode == 'ese_box_center':
        center_x, center_y, center_z = (instance.shape[0] - 1) / 2, (instance.shape[1] - 1) / 2, (instance.shape[2] - 1) / 2
    else:
        center_x, center_y, center_z = 0, 0, 0
    edges = get_gradient(instance)  # your implementation of get gradient, it is a bool map
    index_d, index_h, index_w = np.where(edges == 1)
    edgepoints_array = np.array([(index_w[i], index_h[i], index_d[i]) for i in range(len(index_h))])
    centerpoints_array = np.array([center_x, center_y, center_z])
    # distance_all = distance.cdist(edgepoints_array,centerpoints_array,'euclidean')

    edgeDict = {}
    # generate empty list for all the angle
    for i in range(0, 360, verse):
        for j in range(0, 181, verse):
            edgeDict[(i, j)] = []

    # The current center cannot be found
    if center_x == 0 and center_y == 0 and center_z == 0:
        return edgeDict, center_x, center_y, center_z

    num1 = 0
    num2 = 0
    num3 = 0
    for i in range(len(index_h)):
        # calculate the degree based on center point
        # clockwise
        # i want to get a deg section of each points
        phi1, theta1 = calculate_spherical_angle(index_h[i], index_w[i], index_d[i], center_y, center_x, center_z)
        phi2, theta2 = calculate_spherical_angle(index_h[i], index_w[i], index_d[i] - 1, center_y, center_x,
                                                 center_z)
        phi3, theta3 = calculate_spherical_angle(index_h[i], index_w[i] - 1, index_d[i], center_y, center_x,
                                                 center_z)
        phi4, theta4 = calculate_spherical_angle(index_h[i] - 1, index_w[i], index_d[i], center_y, center_x,
                                                 center_z)
        phi5, theta5 = calculate_spherical_angle(index_h[i], index_w[i], index_d[i] + 1, center_y, center_x,
                                                 center_z)
        phi6, theta6 = calculate_spherical_angle(index_h[i], index_w[i] + 1, index_d[i], center_y, center_x,
                                                 center_z)
        phi7, theta7 = calculate_spherical_angle(index_h[i] + 1, index_w[i], index_d[i], center_y, center_x,
                                                 center_z)

        phi_1 = min(phi1, phi2, phi3, phi4, phi5, phi6, phi7)
        phi_2 = max(phi1, phi2, phi3, phi4, phi5, phi6, phi7)
        theta_1 = min(theta1, theta2, theta3, theta4, theta5, theta6, theta7)
        theta_2 = max(theta1, theta2, theta3, theta4, theta5, theta6, theta7)
        # calculate distance
        dot_array = np.array([index_w[i], index_h[i], index_d[i]])
        distance_r = np.linalg.norm(dot_array - centerpoints_array)

        if int(theta_2) > 176 or int(theta_1) < 4:
            num1 += 1
            for theta in range(max(round_to_next_down(theta_1), 0), round_to_next_up(theta_2),
                               verse):
                for phi in range(0, 360, verse):
                    edgeDict[(phi, theta)].append(distance_r)
        elif int(theta_2) > 165 or int(theta_1) < 15:
            num2 += 1
            if int(phi_2 - phi_1) > 200:
                for theta in range(round_to_next_down(theta_1), round_to_next_up(theta_2), verse):
                    for phi in range(0, min(round_to_next_up(phi_1), 360), verse):
                        edgeDict[(phi, theta)].append(distance_r)
                    for phi in range(round_to_next_down(phi_2), 360, verse):
                        edgeDict[(phi, theta)].append(distance_r)
            else:
                for theta in range(round_to_next_down(theta_1), round_to_next_up(theta_2), verse):
                    for phi in range(round_to_next_down(phi_1), min(round_to_next_up(phi_2), 360), verse):
                        edgeDict[(phi, theta)].append(distance_r)
        else:
            num3 += 1
            if int(phi_2 - phi_1) > 200:
                for theta in range(round_to_next_down(theta_1), round_to_next_up(theta_2), verse):
                    for phi in range(0, min(round_to_next_up(phi_1), 360), verse):
                        edgeDict[(phi, theta)].append(distance_r)
                    for phi in range(round_to_next_down(phi_2), 360, verse):
                        edgeDict[(phi, theta)].append(distance_r)
            else:
                for theta in range(round_to_next_down(theta_1), round_to_next_up(theta_2), verse):
                    for phi in range(round_to_next_down(phi_1), min(round_to_next_up(phi_2), 360), verse):
                        edgeDict[(phi, theta)].append(distance_r)
    # sorted method
    # edgeDict = {k:sorted(edgeDict[k]) for k in edgeDict.keys()}
    start_deg = 0
    '''
    change start_points
    '''
    # find the largest r for each deg
    # print(num1, num2, num3)        # 20 93 2331

    try:

        edgeDict = {(i, j): np.max(np.array(edgeDict[(i, j)])) for (i, j) in edgeDict.keys()}

    except ValueError:
        print("Value Error")

        for index_phi in range(0, 360, verse):
            for index_theta in range(0, 181, verse):
                if len(edgeDict[(index_phi, index_theta)]) == 0:
                    print(index_phi, index_theta)
                    search_phi = (index_phi - verse * 2) % 360
                    search_theta = (index_theta - verse * 2) % (180 + verse)
                    num1 = 0
                    num2 = 0
                    while len(edgeDict[(search_phi % 360, search_theta % (180 + verse))]) == 0:
                        print(search_phi, search_theta)
                        if num1 < 5:
                            search_phi += verse
                            search_phi = search_phi % 360
                            num1 += 1
                        elif num2 < 5:
                            search_theta += verse
                            search_theta = search_theta % (180 + verse)
                            num1 = 0
                            num2 += 1
                        else:
                            break
                    search_info = edgeDict[(search_phi, search_theta)]

                    if len(search_info) != 0:
                        for r_info in search_info:
                            assisPolar = (search_phi % 360, search_theta % (180 + verse), r_info)
                            center_coord = (center_x, center_y, center_z)
                            trans_r = trans_polarone_to_another(index_phi, index_theta, assisPolar, center_coord, instance.shape)
                            edgeDict[(index_phi, index_theta)].append(trans_r)
                    else:
                        edgeDict[(index_phi, index_theta)].append(0)

        edgeDict = {(i, j): np.max(np.array(edgeDict[(i, j)])) for (i, j) in edgeDict.keys()}

    points = [[edgeDict[(phi_num, theta_num)] for theta_num in range(0, 181, verse)] for phi_num in range(0, 360, verse)]

    return points, center_x, center_y, center_z


def round_to_next_up(number):
    """Get the nearest multiple of verse that is greater than it """
    next_up = math.ceil(number / verse) * verse + 1
    return next_up


def round_to_next_down(number):
    """Get the nearest multiple of verse less than it"""
    next_down = math.floor(number / verse) * verse
    return next_down


def trans_polarone_to_another(ori_phi, ori_theta, assisPolar, center_coord, im_shape):
    """
    make sure that the r,theta you want to assis not outof index
    assisPolar = (phi, theta, r_info)
    center_coord = (center_x, center_y, center_z)
    """
    assis_r = np.array(assisPolar[2], np.float32)
    ori_phi = np.array(ori_phi, np.float32)
    ori_theta = np.array(ori_theta, np.float32)
    ori_r = assisPolar[2]

    x = -1
    y = -1
    z = -1

    while not (0 <= x < im_shape[2] and 0 <= y < im_shape[1] and 0 <= z < im_shape[0]):
        x, y, z = spherical_to_cartesian(assis_r, ori_theta, ori_phi)
        x += center_coord[0]
        y += center_coord[1]
        z += center_coord[2]
        x = int(x)
        y = int(y)
        z = int(z)
        ori_r = assis_r
        assis_r -= 0.1
    return ori_r


# center, bbox, r should be converted into arrays, where r is a two-dimensional array
def decoding_theta(data, dim):
    """Convert polar coordinates within a given bounding box and map it to a rectangular coordinate system"""
    centers = np.array(list(data['center'])).astype(np.float32)
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
    center_xs = centers[0]  # 1
    center_ys = centers[1]  # 1
    center_zs = centers[2]  # 1
    idx = np.linspace(0, 360, dim[0], endpoint=False).astype(np.int32).reshape(dim[0], 1)
    idy = np.linspace(0, 180 + verse, dim[1], endpoint=False).astype(np.int32).reshape(dim[1], 1)
    # represents the length of each direction vector
    rs = np.array(data['r'])
    print(rs.shape)
    # rs = np.float32(data['r'])[idx][idy]
    rs = rs.astype(np.float32)
    idx_list = np.flip(idx, axis=0).astype(np.float32)
    idy_list = np.flip(idy, axis=0).astype(np.float32)
    phi_list = idx * np.ones((1, dim[1])).astype(np.float32)
    theta_list = idy.T * np.ones((dim[0], 1)).astype(np.float32)
    # Convert polar coordinates to rectangular coordinates
    # print('theta: ', theta_list[:, 0], theta_list[0, :])
    # print('phi: ', phi_list[:, 0], phi_list[0, :])
    y, x, z = spherical_to_cartesian(rs, theta_list, phi_list)
    # the contour coordinates after adding the offset of center
    x1 = x + center_xs.astype(np.float32)
    y1 = y + center_ys.astype(np.float32)
    z1 = z + center_zs.astype(np.float32)
    r1 = x**2 + y**2 + z**2
    polygons = np.stack((x1, y1, z1), axis=1)
    # print('polygons: ', polygons.shape)
    # x1:  (72, 37)
    # polygons:  (72, 3, 37)
    return polygons, x1, y1, z1, r1


def bbox(mask):
    """Calculate the bounding box of the given binary mask."""
    index = np.nonzero(mask)
    index = np.transpose(index)
    x_min = np.min(index[:, 0])
    x_max = np.max(index[:, 0])
    y_min = np.min(index[:, 1])
    y_max = np.max(index[:, 1])
    z_min = np.min(index[:, 2])
    z_max = np.min(index[:, 2])
    x_len = x_max - x_min
    y_len = y_max - y_min
    z_len = z_max - z_min
    return x_len, y_len, z_len


def get_filelist(path):
    """Retrieves a list of files in the specified directory"""
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(filename)
    return Filelist


def MarchingCubes(image, threshold):
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.ComputeNormalsOn()
    mc.ComputeGradientsOn()
    mc.SetValue(0, threshold)
    mc.Update()


if __name__ == "__main__":
    img_folder = 'D:/xxxxxxx/'

    verse = 5
    dim = (int(360/verse), int(180/verse + 1))
    print(dim)         # (72, 37)

    img_file_name = "sub-verse070_ct.nii.gz_gt.nii.gz"
    # Get file list
    img_file_path = os.path.join(img_folder, img_file_name)

    img = sitk.ReadImage(img_file_path)

    img_array = sitk.GetArrayFromImage(img)

    unique_labels = sorted(list(set(img_array[img_array != 0])))
    # The value of the mask needs to be binarized

    '''
    # colors = np.linspace(0.25, 0.75, len(unique_labels))
    # Generate sample data
    shape = (len(unique_labels), 3)  # 10x10 grid, 3 channels (RGB)
    rgb_array = np.zeros(shape)
    # Create a gradient color
    for i in range(shape[0]):
        rgb_array[i, 0] = 1.0  # Red channel
        rgb_array[i, 1] = i / (shape[0] - 1)  # Glue channel
        rgb_array[i, 2] = 1.0  # Blue channel
    '''
    # Define the start and end colors of the gradient
    start_color = np.array([0, 1, 0])  # Green
    end_color = np.array([1, 1, 0])  # Yellow

    # Define the shape of the array
    array_shape = (len(unique_labels), 3)  # 100 points in the gradient, each with 3 RGB channels

    # Generate gradient array
    gradient_array = np.linspace(start_color, end_color, array_shape[0])

    # mlab.figure(size=(1000, 1000))
    index = 0
    for label in unique_labels:
        # Find the indices of all points equal to the current label value
        img_mask = (img_array == label).astype(np.uint8)

        center = centerdot3D(img_mask)                  # Center index under full image size
        gradient_volume = get_gradient(img_mask)
        print('shape: ', gradient_volume.shape)            # Image original scale   shape:  (356, 356, 73)
        print('label: ', label)
        points, center_x, center_y, center_z = getOrientedPoints(img_mask, verse, dim)

        print('center: ', center_z, center_y, center_x)
        print('size of points: ', len(points), len(points[0]))    # size of points:  72 37
        points_array = np.array(points)
        print('array shape', points_array.shape)

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
        mlab.pipeline.volume(src, color=tuple(gradient_array[index]))
        # mlab.pipeline.volume(src, color=(0.8, 0.8, 0.8))  # Use gray colormap
        mlab.points3d(center[2], center[1], center[0], color=(1, 0, 0), scale_factor=1.5)

        points_array = np.array(out)
        mlab.points3d(points_array[:, 2], points_array[:, 1], points_array[:, 0], color=(0, 0.54, 0.54), scale_factor=0.7)

        # Generate mesh
        # mesh = mlab.mesh(zr, yr, xr, scalars=r1)
        '''

        # Set interactive mode
        mlab.view(azimuth=0, elevation=90, distance=100)
        # mlab.show()

        # Set the scene's perspective using the scene object's engine property.
        engine = mlab.get_engine()
        scene = engine.scenes[0]

        # Set perspective
        scene.scene.camera.position = [100, 100, 100]
        scene.scene.camera.focal_point = [0, 0, 0]
        scene.scene.camera.view_up = [1, 1, 1]
        # Keep a handle to the graphics window
        previous_figure = mlab.gcf()
        # Plot data using mlab.contour3d
        # mlab.contour3d(x, y, z, volume, contours=10, opacity=0.5)
        '''
        index += 1
        # show scene
        mlab.show(stop=False)             # stop=False
        









import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from scipy import ndimage
import random
import itk
import scipy

def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(filename)
    return Filelist


def ImageResample(sitk_image, new_spacing=[1.0, 1.0, 1.0], is_label=False):
    '''
    sitk_image:
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''

    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_spacing = np.array(new_spacing)
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]  # 这样 spacing 还会又更小的小数，但感觉肯能是有必要的
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetOutputPixelType(sitk.sitkUInt8)

        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetOutputPixelType(sitk.sitkFloat32)
        resample.SetInterpolator(sitk.sitkBSpline)  # 83s
        # resample.SetInterpolator(sitk.sitkLinear)  # 1s
        # resample.SetInterpolator(sitk.sitkBSpline1)  # 90s
        # resample.SetInterpolator(sitk.sitkBSpline3)  # 106s


    # resample.SetTransform(sitk.Euler3DTransform())
    newimage = resample.Execute(sitk_image)
    return newimage, new_spacing_refine



def npz2nii(root_path, file, new_space, wirte_path):

    img_file_path = os.path.join(root_path, file)

    img = sitk.ReadImage(img_file_path)

    origin = np.array(img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(img.GetSpacing())
    direction = np.array(img.GetDirection())

    print('original origin, spacing, direction: ', origin, spacing, direction)
    # resample
    if 'img' in file:
        img, new_spacing = ImageResample(img, new_space, is_label=False)
    else:
        img, new_spacing = ImageResample(img, new_space, is_label=True)
        img = sitk.GetArrayFromImage(img)
        img[img < 0.0] = 0.0
        img[img > 25.5] = 0.0
        img = sitk.GetImageFromArray(img)


    print('new_space: ', new_spacing)



    img.SetOrigin(origin)
    img.SetSpacing(new_spacing)
    img.SetDirection(direction)

    sitk.WriteImage(img, wirte_path + file)



if __name__ == "__main__":
    wirte_path = 'xxxxxxxxx/'
    root_folder = "xxxxxxxxxxxxxx/"

    # img_folder = root_folder + 'img/'
    # label_folder = root_folder + 'label/'
    Filelist = get_filelist(root_folder)
    print(len(Filelist))

    new_space = np.array([2.0, 1.0, 1.0])

    for file in Filelist:
        npz2nii(root_folder, file, new_space, wirte_path)
        print(file)


















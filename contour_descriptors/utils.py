import os
import numpy as np
import math
import random

root_dir = os.getcwd()
# input im hxw     im[:,:,0]
# add 2
def add_edge(im):
    """edge extension"""
    d, h, w = im.shape[0], im.shape[1], im.shape[2]
    # add_edge_im = np.zeros((h+10,w+10))
    # add_edge_im[5:h+5,5:w+5] = im
    add_edge_im = im
    return add_edge_im


def get_gradient(im):
    """Compute the gradient map of an input image."""
    d, h, w = im.shape[0], im.shape[1], im.shape[2]
    # Perform edge expansion, adding a ring of pixels around the image to compute gradients at the edges
    im = add_edge(im)
    # Get the unique identifier of the instance
    instance_id = np.unique(im)[1]
    # Create a matrix of zeros the same size as the input image
    mask = np.zeros((im.shape[0], im.shape[1], im.shape[2]))
    # Fill the matrix of zeros with the identifiers of the instances
    mask.fill(instance_id)
    boolmask = (im == mask)
    im = im * boolmask    # only has object

    # Calculate the directional gradient of an image
    z = np.gradient(im)[0]
    y = np.gradient(im)[1]
    x = np.gradient(im)[2]
    gradient = abs(x) + abs(y) + abs(z)
    bool_gradient = gradient.astype(bool)
    mask.fill(1)
    gradient_map = mask * bool_gradient * boolmask
    # gradient_map = gradient_map[5:h+5,5:w+5]
    return gradient_map


def cartesian_to_spherical(x, y, z):
    """Convert the Cartesian coordinates of a vector to spherical coordinates"""
    # Calculate radius r
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Calculate polar angle theta
    if r == 0:
        theta = 0.0
    else:
        theta = np.arccos(z / r)

    # Calculate the azimuth angle phi
    phi = np.arctan2(y, x)

    return r, theta, phi


def calculate_spherical_angle(x, y, z, center_x, center_y, center_z):
    """Convert Cartesian coordinates of a point to spherical coordinates"""
    r, theta, phi = cartesian_to_spherical(x - center_x, y - center_y, z - center_z)

    # Calculate angle
    phi = phi * 180 / np.pi
    if phi < 0:
        phi += 360

    theta = theta * 180 / np.pi

    return phi, theta


def spherical_to_cartesian(radius, theta, phi):
    """Convert the spherical coordinates of a point to Cartesian coordinates"""
    # print('r, theta, phi shape:', radius.shape, theta.shape, phi.shape)
    # r, theta, phi shape: (72, 37) (72, 37) (72, 37)

    theta = np.radians(theta)
    phi = np.radians(phi)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return x, y, z

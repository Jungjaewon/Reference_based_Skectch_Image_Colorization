import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import thinplate as tps
import cv2
import random

def show_warped(img, warped, c_src, c_dst):
    fig, axs = plt.subplots(1, 2, figsize=(16,8))
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].imshow(img[...,::-1], origin='upper')
    axs[0].scatter(c_src[:, 0]*img.shape[1], c_src[:, 1]*img.shape[0], marker='+', color='black')
    axs[1].imshow(warped[...,::-1], origin='upper')
    axs[1].scatter(c_dst[:, 0]*warped.shape[1], c_dst[:, 1]*warped.shape[0], marker='+', color='black')
    plt.show()

def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

if __name__ == '__main__':
    img = cv2.imread('1000000_color.png')


    while True:
        point1 = round(random.uniform(0.3, 0.7), 2)
        point2 = round(random.uniform(0.3, 0.7), 2)
        range_1 = round(random.uniform(-0.25, 0.25), 2)
        range_2 = round(random.uniform(-0.25, 0.25), 2)
        if point1 + range_1 == point2 + range_2:
            continue
        else:
            break

    c_src = np.array([
        [0.0, 0.0],
        [1., 0],
        [1, 1],
        [0, 1],
        [point1, point1],
        [point2, point2],
    ])

    c_dst = np.array([
        [0., 0],
        [1., 0],
        [1, 1],
        [0, 1],
        [point1 + range_1, point1 + range_1],
        [point2 + range_2, point2 + range_2],
    ])

    warped = warp_image_cv(img, c_src, c_dst, dshape=(512, 512))
    show_warped(img, warped, c_src, c_dst)


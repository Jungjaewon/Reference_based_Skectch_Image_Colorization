from PIL import Image,  ImageEnhance
from pylab import *
from scipy.ndimage import filters
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# Python tps https://github.com/cheind/py-thin-plate-spline/blob/master/thinplate/pytorch.py
# Reference : https://hobbydev.tistory.com/56
# https://github.com/WarBean/tps_stn_pytorch tps stn pytorch

def get_outline_image(img_path):
    gamma = 0.97
    phi = 200
    epsilon = 0.1
    k = 2.5
    sigma = 0.7
    im = Image.open(img_path).convert('L')
    im = array(ImageEnhance.Sharpness(im).enhance(3.0))
    im2 = filters.gaussian_filter(im, sigma)
    im3 = filters.gaussian_filter(im, sigma * k)
    differencedIm2 = im2 - (gamma * im3)
    (x, y) = shape(im2)
    for i in range(x):
        for j in range(y):
            if differencedIm2[i, j] < epsilon:
                differencedIm2[i, j] = 1
            else:
                differencedIm2[i, j] = 250 + tanh(phi * (differencedIm2[i, j]))

    gray_pic = differencedIm2.astype(np.uint8)
    final_img = Image.fromarray(gray_pic)
    return final_img

# https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    print('image : ', id(image))
    if random_state is None:
        random_state = np.random.RandomState(None)

        # print(random_state)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    distored_image = map_coordinates(image, indices, order=1, mode='nearest')  # wrap,reflect, nearest
    return distored_image.reshape(image.shape)

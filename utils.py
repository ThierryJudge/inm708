import scipy
import numpy as np
from skimage.filters import median
from skimage.restoration import denoise_bilateral
import operator
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

def michelson_contrast(im):
    return (np.max(im) - np.min(im)) / (np.max(im) + np.max(im))


def rms_contrast(im):
    mean = np.mean(im)
    return np.sqrt(1 / (np.prod(im.shape) - 1) * np.sum(np.square(im - mean)))


def cropND(img, start, bounding):
    """https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image"""
    end = tuple(map(operator.add, start, np.repeat(bounding, img.ndim)))
    slices = tuple(map(slice, start, end))
    return img[slices]


def SNR(im, fg_coord, bg_coord, window_size):
    assert len(fg_coord) == im.ndim, "FG window coordinates must be same lenght as image dimensions"
    assert len(bg_coord) == im.ndim, "BG window coordinates must be same lenght as image dimensions"

    fg_coord = np.array(fg_coord)
    bg_coord = np.array(bg_coord)

    assert not np.any((fg_coord + window_size) > im.shape), 'Window must be in image'
    assert not np.any((bg_coord + window_size) > im.shape), 'Window must be in image'
    assert not np.any(fg_coord < 0), 'Window must be in image'
    assert not np.any(bg_coord < 0), 'Window must be in image'

    fg = cropND(im, fg_coord, window_size)
    bg = cropND(im, bg_coord, window_size)

    return np.mean(fg)/ (np.std(bg) + 1e-8)


def gaussian_filter(im, sigma):
    return scipy.ndimage.gaussian_filter(im, sigma=sigma)


def median_filter(im):
    return median(im)


def bilateral_filter(im):
    return denoise_bilateral(im)


def nlmeans_filter(im, mask_value, patch_radius=1, block_radius=1, rician=True):
    mask = im > mask_value
    sigma = estimate_sigma(im, N=0)

    den = nlmeans(im, sigma=sigma, mask=mask, patch_radius=patch_radius,
                  block_radius=block_radius, rician=rician)

    return den

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

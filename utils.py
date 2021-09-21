import scipy
import numpy as np
from skimage.filters import median
from skimage.restoration import denoise_bilateral


def michelson_contrast(im):
    if im.ndim != 2:
        raise Exception("Input must be a 2D image")

    return (np.max(im) - np.min(im)) / (np.max(im) + np.max(im))


def rms_contrast(im):
    if im.ndim != 2:
        raise Exception("Input must be a 2D image")

    it = 0
    m_ = np.mean(im)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            it += (im[i][j] - m_) ** 2

    return np.sqrt(1 / (im.shape[0] * im.shape[1] - 1) * it)


def SNR(im, S, fond, window_size):
    if im.ndim != 2:
        raise Exception("Input must be a 2D image")

    if not isinstance(S, tuple) or len(S) != 2:
        raise Exception("S window starting point must be defined as a tuple of "
                        "len 2")

    if not isinstance(fond, tuple) or len(fond) != 2:
        raise Exception("fond window starting point must be defined as a tuple "
                        "of len 2")

    if S[0] + window_size > im.shape[0] or\
            S[1] + window_size > im.shape[1]:
        raise Exception("Combination of window size and starting pixel of S "
                        "exceed image dimension")
    if fond[0] + window_size > im.shape[0] or\
            fond[1] + window_size > im.shape[1]:
        raise Exception("Combination of window size and starting pixel of "
                        "\"fond\" exceed image dimension")

    S_window = im[S[0]:S[0]+window_size, S[1]:S[1]+window_size]
    fond_window = im[fond[0]:fond[0]+window_size, fond[1]:fond[1]+window_size]

    return np.mean(S_window)/np.std(fond_window)


def gaussian_filter(im, s):
    return scipy.ndimage.gaussian_filter(im, sigma=s)


def median_filter(im):
    return median(im)


def bilateral_filter(im):
    return denoise_bilateral(im)


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
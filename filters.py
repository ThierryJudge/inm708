import scipy
import numpy as np
from skimage.filters import median
from skimage.restoration import denoise_bilateral
import operator
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from medpy.filter.smoothing import anisotropic_diffusion


def gaussian_filter(im, sigma):
    return scipy.ndimage.gaussian_filter(im, sigma=sigma)


def median_filter(im):
    return median(im)


def bilateral_filter(im):
    return denoise_bilateral(im)


def anisotropic_diffusion_filter(img):
    filtered = anisotropic_diffusion(img)
    return filtered


def nlmeans_filter(im, mask_value, patch_radius=1, block_radius=1, rician=True):
    mask = im > mask_value
    sigma = estimate_sigma(im, N=0)

    den = nlmeans(im, sigma=sigma, mask=mask, patch_radius=patch_radius,
                  block_radius=block_radius, rician=rician)

    return den

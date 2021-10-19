import numpy as np


def flatten_rem_black(im1, im2, n):
    assert im1.shape == im2.shape

    return np.sort(im1.flatten())[n:np.prod(im1.shape)], np.sort(
        im2.flatten())[n:np.prod(im2.shape)]


def pretty_print(s, value):
    print(s + (30 - len(s)) * " " + f"{value}")


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

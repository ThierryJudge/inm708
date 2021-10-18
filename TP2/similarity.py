import numpy as np
from pathlib import Path

import numpy as np
import imageio
from matplotlib import pyplot as plt

from histogram import histogram


def ssd(i, j):
    return np.sum(np.square(i - j))


def cr(i, j):
    # np.cov returns 2x2 matrix, get second (or third) value to get correlation between i and j
    # return np.cov(i.ravel(), j.ravel()).ravel()[1] / (np.var(i) * np.var(j))
    return np.corrcoef(i.ravel(),j.ravel()).ravel()[1]


def mi(i, j, bins=100):
    hist = histogram(i, j, bins)
    hist_norm = hist / np.sum(hist)
    px = np.sum(hist_norm, axis=1)
    py = np.sum(hist_norm, axis=0)
    pxpy = px[None, :] * py[:, None]

    indices = hist_norm > 0  # Only consider values that are over 0 to avoid NaN errors with log
    return np.sum(hist_norm[indices] * np.log(hist_norm[indices] / pxpy[indices]))

    # return np.sum(hist_norm * np.log(hist_norm / pxpy))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help='Path to the data folder')
    parser.add_argument("--bins", type=int, help='Number of bins for histogram', default=100)
    args = parser.parse_args()

    for i in range(1, 7):
        try:
            I = imageio.imread(args.path / f'I{i}.png')
            J = imageio.imread(args.path / f'J{i}.png')
        except:
            I = imageio.imread(args.path / f'I{i}.jpg')
            J = imageio.imread(args.path / f'J{i}.jpg')

        print(f'Paire {i}: SSD : {ssd(I, J)}, Corr.: {cr(I, J)}, MI: {mi(I, J, args.bins)}')




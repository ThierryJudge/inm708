from pathlib import Path

import numpy as np
import imageio
from matplotlib import pyplot as plt


def histogram(i, j, bins=10):
    H, _, _ = np.histogram2d(i.ravel(), j.ravel(), bins=bins)
    return H


if __name__ == '__main__':
    import argparse
    from similarity import ssd, cr, mi

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help='Path to the data folder')
    parser.add_argument("--bins", type=int, help='Number of bins', default=100)
    parser.add_argument("--log_scale", action='store_true', help='Use log scale')
    args = parser.parse_args()

    for i in range(1, 7):
        try:
            I = imageio.imread(args.path / f'I{i}.png')
            J = imageio.imread(args.path / f'J{i}.png')
        except:
            I = imageio.imread(args.path / f'I{i}.jpg')
            J = imageio.imread(args.path / f'J{i}.jpg')

        assert I.shape == J.shape

        hist = histogram(I, J, args.bins)

        print(f"Histogram sum {np.sum(hist)}, Number of pixels: {np.prod(I.shape)}")
        assert np.sum(hist) == np.prod(I.shape)

        if args.log_scale:
            hist = np.log(hist)

        print(f'Paire {i}: SSD : {ssd(I, J)}, Corr.: {cr(I, J)}, MI: {mi(I, J, args.bins)}')

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(I)
        ax1.set_title(f"I{i}")
        ax2.imshow(J)
        ax2.set_title(f"J{i}")
        ax3.imshow(hist, origin='lower')
        ax3.set_title("H")
        plt.show()

from pathlib import Path

import numpy as np
import imageio
from matplotlib import pyplot as plt
import seaborn as sns

from utils import *


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
    parser.add_argument("--remove_black", type=int, help='Use seaborn?')
    args = parser.parse_args()

    print( "\n" + bcolors.OKBLUE + "Informations: " + bcolors.ENDC + "\n")

    for i in range(1, 7):
        try:
            I = imageio.imread(args.path / f'I{i}.png')
            J = imageio.imread(args.path / f'J{i}.png')
        except:
            I = imageio.imread(args.path / f'I{i}.jpg')
            J = imageio.imread(args.path / f'J{i}.jpg')

        assert I.shape == J.shape

        hist = histogram(I, J, args.bins)

        print(bcolors.UNDERLINE + f'\nPaire {i}:' + bcolors.ENDC)
        pretty_print("Histogram sum:", np.sum(hist))
        pretty_print("Number of pixels:", np.prod(I.shape))

        assert np.sum(hist) == np.prod(I.shape)

        if args.log_scale:
            hist = np.log(hist)

        pretty_print("SSD:", ssd(I, J))
        pretty_print("Correlation:", cr(I, J))
        pretty_print("MI:", mi(I, J, args.bins))


        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(I)
        ax1.set_title(f"I{i}")
        ax2.imshow(J)
        ax2.set_title(f"J{i}")
        ax3.imshow(hist, origin='lower')
        ax3.set_title("H")

        if args.remove_black:
            sns.set_theme(style="ticks")

            im1, im2 = flatten_rem_black(I, J,
                                         int(np.prod(I.shape)*args.remove_black/100))
            sns.jointplot(x=im1, y=im2,
                          kind="hex", color="#4CB391",
                          marginal_kws=dict(bins=args.bins,
                                            fill=True))
        plt.show()

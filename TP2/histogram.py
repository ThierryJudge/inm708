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

    print("\n" + bcolors.OKBLUE + "Informations: " + bcolors.ENDC)
    
    num_images = 6
    figure, axes = plt.subplots(num_images, 3, figsize=(4, 8))

    for i in range(1, num_images+1):
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
            hist[abs(hist) == np.inf] = 0

        pretty_print("SSD:", ssd(I, J))
        pretty_print("Correlation:", cr(I, J))
        pretty_print("IM:", mi(I, J, args.bins))

        aspect = None
        if i-1 == 0:
            axes[i - 1, 0].set_title(f"I")
            axes[i - 1, 1].set_title(f"J")
            axes[i - 1, 2].set_title("H")

        axes[i-1, 0].imshow(I, aspect=aspect)
        axes[i - 1, 0].set_ylabel(f'{i}', rotation=90)
        axes[i-1, 1].imshow(J, aspect=aspect)
        plot = axes[i-1, 2].imshow(hist, origin='lower', aspect=aspect)
        plt.colorbar(plot, ax=axes[i-1, 2])

        axes[i-1, 0].set_xticks([])
        axes[i-1, 0].set_yticks([])

        axes[i-1, 1].set_xticks([])
        axes[i-1, 1].set_yticks([])

        axes[i-1, 2].set_xticks([])
        axes[i-1, 2].set_yticks([])

        if args.remove_black:
            sns.set_theme(style="ticks")

            im1, im2 = flatten_rem_black(I, J,
                                         int(np.prod(I.shape)*args.remove_black/100))
            sns.jointplot(x=im1, y=im2,
                          kind="hex", color="#4CB391",
                          marginal_kws=dict(bins=args.bins,
                                            fill=True))

    figure.savefig('histogram.png')
    plt.show()

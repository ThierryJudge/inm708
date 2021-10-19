from pathlib import Path

import imageio
import numpy as np
from matplotlib import pyplot as plt
from optimizer import GradientDescent, Optimizer, Adam
from scipy.ndimage import affine_transform
from similarity import ssd
from tqdm import tqdm

from recalage import rigid, rotate, recalage_rigide

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", type=Path, help='Path to source image')
    parser.add_argument("target_path", type=Path, help='Path to target image')
    args = parser.parse_args()

    I = imageio.imread(args.source_path)
    J = imageio.imread(args.target_path)

    optim = Adam(eps=1e-2)

    variables, errors = recalage_rigide(I, J, optim, 10000)
    optim.plot_variables(names=['p', 'q', 'theta'])
    Ir = rigid(I, *tuple(-variables))

    plt.figure()
    plt.plot(errors)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(J)
    ax1.set_title("Target")
    ax2.imshow(I)
    ax2.set_title("Initial")
    ax3.imshow(Ir)
    ax3.set_title("Registered")

    plt.show()
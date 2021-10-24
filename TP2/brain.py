from pathlib import Path

import imageio
from matplotlib import pyplot as plt
import os
from optimizer import Adam
from recalage import rigid, recalage_rigide
import numpy as np

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", type=Path, help='Path to source image')
    parser.add_argument("target_path", type=Path, help='Path to target image')
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eps", type=float, default=1e-2)
    args = parser.parse_args()

    I = imageio.imread(args.source_path)
    J = imageio.imread(args.target_path)

    optim = Adam(eps=args.eps)

    variables, errors = recalage_rigide(I, J, optim, args.steps)
    variables = optim.variable_history[np.argmin(errors)]
    Ir = rigid(I, *tuple(-variables))

    names = ['p', 'q', 'theta']
    optim.variable_history[:, 2] = np.rad2deg(optim.variable_history[:, 2])

    figure1, axes = plt.subplots(1, len(names) + 1)
    axes = axes.ravel()

    axes[0].set_title("Error")
    axes[0].plot(errors)

    for i in range(len(names)):
        axes[i + 1].set_title(names[i])
        axes[i + 1].plot(optim.variable_history.T[i])

    figure2, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.imshow(J)
    ax1.set_title("Target")
    ax2.imshow(I)
    ax2.set_title("Initial")
    ax3.imshow(Ir)
    ax3.set_title("Registered")
    ax4.imshow(J - Ir)
    ax4.set_title("Difference")


    spath = os.path.splitext(os.path.basename(args.source_path))[0]
    tpath = os.path.splitext(os.path.basename(args.target_path))[0]
    figure1.savefig(f"{spath}{tpath}_curve.png")
    figure2.savefig(f"{spath}2{tpath}_ex.png")

    plt.show()

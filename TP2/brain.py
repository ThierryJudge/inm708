from pathlib import Path

import imageio
from matplotlib import pyplot as plt

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
    Ir = rigid(I, *tuple(-variables))

    names = ['p', 'q', 'theta']
    optim.variable_history[:, 2] = np.rad2deg(optim.variable_history[:, 2])

    figure, axes = plt.subplots(1, len(names) + 1)
    axes = axes.ravel()

    axes[0].set_title("Error")
    axes[0].plot(errors)
    axes[0].legend()

    for i in range(len(names)):
        axes[i + 1].set_title(names[i])
        axes[i + 1].plot(optim.variable_history.T[i])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(J)
    ax1.set_title("Target")
    ax2.imshow(I)
    ax2.set_title("Initial")
    ax3.imshow(Ir)
    ax3.set_title("Registered")

    plt.show()
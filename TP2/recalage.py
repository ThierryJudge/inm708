from collections.abc import Sequence
from pathlib import Path
from typing import Tuple

import imageio
from scipy.ndimage import affine_transform
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from similarity import ssd


def translate(img, p, q):
    mat = np.eye(3, 3)
    mat[0, 2] = p
    mat[1, 2] = q

    return affine_transform(img, mat)


# class Optimizer:
#     def __init__(self, eps: float):
#         self.eps = eps
#
#     def step(self, variables: Sequence, grads: Sequence) -> Tuple:
#         raise NotImplementedError
#
#
# class GD(Optimizer):
#
#     def step(self, variables: np.ndarray, grads: np.ndarray) -> np.ndarray:
#         variables = variables + self.eps * grads
#         return variables
#
#
# class GD_Momentum(Optimizer):
#     def __init__(self, eps: float, momentum: float):
#         super().__init__(eps)
#         self.momentum = momentum
#         self.v = None
#
#     def step(self, variables: np.ndarray, grads: np.ndarray) -> np.ndarray:
#         if self.v is None:
#             self.v = np.zeros(len(variables))
#
#         self.v = self.momentum * self.v + self.eps * g
#         variables = variables + self.v
#
#         return variables


def recalage_translation(i, j, iterations=500, eps=1e-9):
    errors = []
    dps, dqs = [], []
    ps, qs = [], []
    p, q = 1, 1
    for _ in tqdm(range(iterations)):
        i_trans = translate(i, -p, -q)
        errors.append(ssd(i_trans, j))

        # dx = np.diff(i_trans, axis=0, prepend=0)
        # dy = np.diff(i_trans, axis=1, prepend=0)

        dx = np.gradient(i_trans, axis=0)
        dy = np.gradient(i_trans, axis=1)

        # dx = translate(dx, p, q)
        # dy = translate(dy, p, q)

        dp = 2 * np.sum((i_trans - j) * dx)
        dq = 2 * np.sum((i_trans - j) * dy)

        # print(p, q)
        # print(dp, dq)

        p -= eps * dp
        q -= eps * dq

        ps.append(p)
        qs.append(q)
        dps.append(dp)
        dqs.append(dq)

        # plt.figure()
        # plt.imshow(i_trans - i)
        # plt.show()

    print(p, q)

    plt.figure()
    plt.plot(ps)
    plt.plot(qs)

    plt.figure()
    plt.plot(dps)
    plt.plot(dqs)

    return p, q, errors


def rotate(img, theta):
    mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    print(mat)

    return affine_transform(img, mat)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help='Path to input image')
    args = parser.parse_args()

    I = imageio.imread(args.path)
    I2 = translate(I, 20, 10)

    p, q, errors = recalage_translation(I2, I, 1000, eps=1e-8)

    I3 = translate(I2, -p, -q)

    plt.figure()
    plt.plot(errors)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(I)
    ax2.imshow(I2)
    ax3.imshow(I3)

    plt.show()

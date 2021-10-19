from pathlib import Path

import imageio
import numpy as np
from matplotlib import pyplot as plt
from optimizer import GradientDescent, Optimizer, Adam
from scipy.ndimage import affine_transform
from similarity import ssd
from tqdm import tqdm


def recalage_translation(i, j, optimizer: Optimizer, iterations=500):
    errors = []
    variables = np.ones(2)  # p, q
    for _ in tqdm(range(iterations)):
        i_trans = translate(i, -variables[0], -variables[1])
        errors.append(ssd(i_trans, j))

        dx = np.gradient(i_trans, axis=0)
        dy = np.gradient(i_trans, axis=1)

        dp = 2 * np.sum((i_trans - j) * dx)
        dq = 2 * np.sum((i_trans - j) * dy)

        gradients = np.array([dp, dq])

        variables = optimizer.step(variables, gradients)

    print(variables)

    return variables, errors


def recalage_rotation(i, j, optimizer: Optimizer, iterations=500):
    errors = []
    variables = np.zeros(1)  # p, q
    for _ in tqdm(range(iterations)):
        i_trans = rotate(i, -variables[0])
        errors.append(ssd(i_trans, j))

        dx = np.gradient(i_trans, axis=0)
        dy = np.gradient(i_trans, axis=1)

        x = np.arange(j.shape[0])
        y = np.arange(j.shape[1])
        sin = np.sin(variables[0])
        cos = np.cos(variables[0])
        dtheta = 2 * np.sum((i_trans - j) * (dx * (-x * sin - y * cos) + dy * (x * cos - y * sin)))

        gradients = np.array([dtheta])

        variables = optimizer.step(variables, gradients)

    print(variables)

    return variables, errors


def recalage_rigide(i, j, optimizer: Optimizer, iterations=500):
    errors = []
    variables = np.zeros(3)  # p, q, theta
    for _ in tqdm(range(iterations)):
        i_trans = rigid(i, *tuple(-variables))
        errors.append(ssd(i_trans, j))

        dx = np.gradient(i_trans, axis=0)
        dy = np.gradient(i_trans, axis=1)

        x = np.arange(j.shape[0])
        y = np.arange(j.shape[1])
        sin = np.sin(variables[2])
        cos = np.cos(variables[2])
        dtheta = 2 * np.sum((i_trans - j) * (dx * (-x * sin - y * cos) + dy * (x * cos - y * sin)))

        dp = 2 * np.sum((i_trans - j) * dx)
        dq = 2 * np.sum((i_trans - j) * dy)

        gradients = np.array([dp, dq, dtheta])

        variables = optimizer.step(variables, gradients)

    print(variables)

    return variables, errors


def translate(img, p, q):
    mat = np.eye(3, 3)
    mat[0, 2] = p
    mat[1, 2] = q
    return affine_transform(img, mat)


def rotate(img, theta):
    mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    return affine_transform(img, mat)


def rigid(img, p, q, theta):
    mat = np.array([[np.cos(theta), -np.sin(theta), p],
                    [np.sin(theta), np.cos(theta), q],
                    [0, 0, 1]])
    return affine_transform(img, mat)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help='Path to input image')
    args = parser.parse_args()

    I = imageio.imread(args.path)
    # I2 = translate(I, 20, 10)
    # I2 = rotate(I, np.deg2rad(10))
    I2 = rigid(I, 20, 10, np.deg2rad(10))

    optim = Adam(eps=1e-4)

    variables, errors = recalage_rigide(I2, I, optim, 10000)
    optim.plot_variables(names=['p', 'q', 'theta'])
    I3 = rigid(I2, *tuple(-variables))

    # variables, errors = recalage_rotation(I2, I, optim, 1000)
    # optim.plot_variables(names=['theta'])
    # I3 = rotate(I2, -variables[0])

    # variables, errors = recalage_translation(I2, I, optim, 1000)
    # optim.plot_variables(names=['p', 'q'])
    # I3 = translate(I2, -variables[0], -variables[1])

    plt.figure()
    plt.plot(errors)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(I)
    ax1.set_title("Initial")
    ax2.imshow(I2)
    ax2.set_title("Transformed")
    ax3.imshow(I3)
    ax3.set_title("Registered")

    plt.show()

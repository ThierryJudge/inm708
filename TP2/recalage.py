import random
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

        x = np.repeat(np.arange(j.shape[0])[None], j.shape[1], axis=0)
        y = np.repeat(np.arange(j.shape[1])[:, None], j.shape[0], axis=1)
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

        x = np.repeat(np.arange(j.shape[0])[None], j.shape[1], axis=0)
        y = np.repeat(np.arange(j.shape[1])[:, None], j.shape[0], axis=1)

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

    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help='Path to input image')
    parser.add_argument("transform", type=str, help='Type of transform', choices=['translation', 'rotation', 'rigid'],
                        default='trans')
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--num_tests", type=int, default=3)

    args = parser.parse_args()

    I = imageio.imread(args.path)

    # Ranges for random rotation and translation
    translation_range = -30, 30
    rotation_range = -np.deg2rad(20), np.deg2rad(20)

    # Used for plotting results
    variable_histories = []
    real_values = []
    images = []
    test_errors = []

    for i in range(args.num_tests):

        print(f"Test recalage f{args.transform} {i} de {args.num_tests}...")

        if args.transform == 'translation':
            names = ['p', 'q']
            optim = GradientDescent(eps=1e-8)

            p = random.uniform(translation_range[0], translation_range[1])
            q = random.uniform(translation_range[0], translation_range[1])
            real_values.append([p, q])

            I2 = translate(I, p, q)

            variables, errors = recalage_translation(I2, I, optim, args.steps)
            variables = optim.variable_history[np.argmin(errors)]

            I3 = translate(I2, -variables[0], -variables[1])

            print(f"Real values: p={p}, q={q}")
            print(f"Computed values: p={variables[0]}, q={variables[1]}")

        elif args.transform == 'rotation':
            names = ['theta']
            optim = GradientDescent(eps=1e-12)

            theta = random.uniform(rotation_range[0], rotation_range[1])

            I2 = rotate(I, theta)

            variables, errors = recalage_rotation(I2, I, optim, args.steps)
            I3 = rotate(I2, -variables[0])

            variables = optim.variable_history[np.argmin(errors)]

            print(f"Real values: theta={np.rad2deg(theta)}")
            print(f"Computed values: theta={np.rad2deg(variables[0])}")

            # Change to degree to facilitate interpretation
            variables[0] = np.rad2deg(variables[0])
            real_values.append([np.rad2deg(theta)])
            optim.variable_history = np.rad2deg(optim.variable_history)

        elif args.transform == 'rigid':
            optim = Adam(eps=1e-3)

            p = random.uniform(translation_range[0], translation_range[1])
            q = random.uniform(translation_range[0], translation_range[1])
            theta = random.uniform(rotation_range[0], rotation_range[1])
            names = ['p', 'q', 'theta']

            I2 = rigid(I, p, q, theta)
            variables, errors = recalage_rigide(I2, I, optim, args.steps)
            variables = optim.variable_history[np.argmin(errors)]
            I3 = rigid(I2, *tuple(-variables))

            print(f"Real values: p={p}, q={q}, theta={np.rad2deg(theta)}")
            print(f"Computed values: p={variables[0]}, q={variables[1]}, theta={np.rad2deg(variables[2])}")

            # Change to degree to facilitate interpretation
            variables[2] = np.rad2deg(variables[2])
            real_values.append([p, q, np.rad2deg(theta)])
            optim.variable_history[:, 2] = np.rad2deg(optim.variable_history[:, 2])

        else:
            raise ValueError("Transform not valid")

        variable_histories.append(optim.variable_history)
        images.append([I, I2, I3])
        test_errors.append(errors)

    figure1, axes = plt.subplots(args.num_tests, 4)
    axes[0, 0].set_title("Initial")
    axes[0, 1].set_title("Transformed")
    axes[0, 2].set_title("Registered")
    axes[0, 3].set_title("Difference")
    for i in range(args.num_tests):
        axes[i - 1, 0].imshow(images[i][0])
        axes[i - 1, 1].imshow(images[i][1])
        axes[i - 1, 2].imshow(images[i][2])
        axes[i - 1, 3].imshow(images[i][0] - images[i][2])

    figure2, axes = plt.subplots(args.num_tests, 2)
    axes[0, 0].set_title("Errors")
    axes[0, 1].set_title("Variables")
    for i in range(args.num_tests):
        axes[i, 0].plot(test_errors[i])
        axes[i, 1].plot(variable_histories[i])
        axes[i, 1].legend(names)

    for i in range(args.num_tests):
        axes[i, 1].set_prop_cycle(None)
        for var in range(len(real_values[i])):
            axes[i, 1].plot([0, args.steps], [real_values[i][var], real_values[i][var]], linestyle='--')

    figure1.savefig(f"recalage_{args.transform}_ex.png")
    figure2.savefig(f"recalage_{args.transform}_curve.png")

    plt.show()

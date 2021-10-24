from recalage import *
import numpy as np

if __name__ == '__main__':
    import argparse

    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help='Path to input image')
    parser.add_argument("transform", type=str, help='Type of transform', choices=['translation', 'rotation', 'rigid'],
                        default='trans')
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--adam_lr", type=float, default=1e-3)
    parser.add_argument("--gd_lr", type=float, default=1e-8)
    parser.add_argument("--gd_mom", type=float, default=0.5)

    args = parser.parse_args()

    I = imageio.imread(args.path)

    optims = [Adam(eps=args.adam_lr),
              GradientDescent(eps=args.gd_lr),
              GradientDescent(eps=args.gd_lr, momentum=args.gd_mom)]

    optim_names = [optim.to_string() for optim in optims]
    print(optim_names)

    # Ranges for random rotation and translation
    translation_range = -30, 30
    rotation_range = -np.deg2rad(20), np.deg2rad(20)

    # Used for plotting results
    variable_histories = []
    real_values = []
    images = []
    test_errors = []

    p = random.uniform(translation_range[0], translation_range[1])
    q = random.uniform(translation_range[0], translation_range[1])
    theta = random.uniform(rotation_range[0], rotation_range[1])

    for optim in optims:

        print(f"Test recalage f{args.transform} avec {optim.__class__.__name__}...")

        if args.transform == 'translation':
            names = ['p', 'q']
            real_values = [p, q]
            I2 = translate(I, p, q)

            variables, errors = recalage_translation(I2, I, optim, args.steps)

            I3 = translate(I2, -variables[0], -variables[1])

            print(f"Real values: p={p}, q={q}")
            print(f"Computed values: p={variables[0]}, q={variables[1]}")

        elif args.transform == 'rotation':
            names = ['theta']
            I2 = rotate(I, theta)

            variables, errors = recalage_rotation(I2, I, optim, args.steps)
            I3 = rotate(I2, -variables[0])

            print(f"Real values: theta={np.rad2deg(theta)}")
            print(f"Computed values: theta={np.rad2deg(variables[0])}")

            # Change to degree to facilitate interpretation
            variables[0] = np.rad2deg(variables[0])
            real_values = [np.rad2deg(theta)]
            optim.variable_history = np.rad2deg(optim.variable_history)

        elif args.transform == 'rigid':

            names = ['p', 'q', 'theta']

            I2 = rigid(I, p, q, theta)

            variables, errors = recalage_rigide(I2, I, optim, args.steps)
            I3 = rigid(I2, *tuple(-variables))

            print(f"Real values: p={p}, q={q}, theta={np.rad2deg(theta)}")
            print(f"Computed values: p={variables[0]}, q={variables[1]}, theta={np.rad2deg(variables[2])}")

            # Change to degree to facilitate interpretation
            variables[2] = np.rad2deg(variables[2])
            real_values = [p, q, np.rad2deg(theta)]
            optim.variable_history[:, 2] = np.rad2deg(optim.variable_history[:, 2])

        else:
            raise ValueError("Transform not valid")

        variable_histories.append(optim.variable_history)
        images.append([I, I2, I3])
        test_errors.append(errors)

    figure1, axes = plt.subplots(len(optims), 4, figsize=(6, 7))
    axes[0, 0].set_title("Initial")
    axes[0, 1].set_title("Transformed")
    axes[0, 2].set_title("Registered")
    axes[0, 3].set_title("Difference")
    for i in range(len(optims)):
        axes[i - 1, 0].imshow(images[i][0])
        axes[i - 1, 1].imshow(images[i][1])
        axes[i - 1, 2].imshow(images[i][2])
        axes[i - 1, 3].imshow(images[i][0] - images[i][2])
        axes[i - 1, 0].set_ylabel(optim_names[i])

    test_errors = np.array(test_errors)
    variable_histories = np.array(variable_histories).transpose((2, 0, 1))

    figure2, axes = plt.subplots(1, len(names) + 1, figsize=(10, 4))
    axes = axes.ravel()

    axes[0].set_title("Errors")
    axes[0].plot(test_errors.T, label=optim_names, alpha=0.5)
    axes[0].legend()
    axes[0].set_xlabel('Steps')

    for i in range(len(names)):
        axes[i + 1].set_title(names[i])
        axes[i + 1].plot(variable_histories[i].T, label=optim_names, alpha=0.5)
        axes[i + 1].axhline(y=real_values[i], linestyle='--', label='Real value', c='r')
        axes[i + 1].legend()
        axes[i + 1].set_xlabel('Steps')

    figure1.savefig(f"compare_optim_{args.transform}_ex.png")
    figure2.savefig(f"compare_optim_{args.transform}_curve.png")
    plt.show()

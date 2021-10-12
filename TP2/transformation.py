import numpy as np
from matplotlib import pyplot as plt


def generate_grid():
    X = np.mgrid[-10:10, -10:10, 0:5]
    X = X.reshape((3, -1)).T
    X = np.concatenate([X, np.ones(len(X))[:, None]], axis=1)
    return X


def rotate_x(theta):
    return np.array([[1., 0., 0., 0],
                     [0., np.cos(theta), -np.sin(theta), 0.],
                     [0., np.sin(theta), np.cos(theta), 0.],
                     [0., 0., 0., 1.]
                     ])


def rotate_y(theta):
    return np.array([[np.cos(theta), 0., np.sin(theta), 0.],
                     [0., 1., 0., 0],
                     [-np.sin(theta), 0, np.cos(theta), 0.],
                     [0., 0., 0., 1.]
                     ])


def rotate_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0., 0.],
                     [np.sin(theta), np.cos(theta), 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]
                     ])


def trans_rigide(theta, omega, phi, p, q, r):
    Ax = rotate_x(theta)
    Ay = rotate_y(omega)
    Az = rotate_z(phi)
    T = np.array([[1., 0., 0, p],
                  [0., 1., 0, q],
                  [0., 0., 0, r],
                  [0., 0., 0., 1.]
                  ])

    return Ay @ Ax


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    X = generate_grid()
    mat = trans_rigide(45., 45., 0., 2., 2., 1.)

    print(X.shape)
    print(mat.shape)

    X2 = np.matmul(X, mat)

    print(X2.shape)

    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c='r')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

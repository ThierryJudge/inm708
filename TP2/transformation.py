import numpy as np
from matplotlib import pyplot as plt


class Grid3D:

    def __init__(self):

        self.g_ = np.mgrid[-10:10, -10:10, 0:5]
        self.g_ = self.g_.reshape((3, -1))
        self.g_ = np.concatenate([self.g_, np.ones(self.g_.shape[-1])[None]],
                                 axis=0)

        print(self.g_.shape)
        print(self.g_)

        self.rotation_X = None
        self.rotation_Y = None
        self.rotation_Z = None
        self.translation_matrix = None
        self.transformation_matrix = None

    def rotate_x(self, theta):
        self.rotation_X = np.array(
            [[1., 0., 0., 0],
             [0., np.cos(theta), -np.sin(theta), 0.],
             [0., np.sin(theta), np.cos(theta), 0.],
             [0., 0., 0., 1.]
             ])

        self.g_ = self.rotation_X @ self.g_

    def rotate_y(self, theta):
        self.rotation_Y = np.array(
            [[np.cos(theta), 0., np.sin(theta), 0.],
             [0., 1., 0., 0],
             [-np.sin(theta), 0, np.cos(theta), 0.],
             [0., 0., 0., 1.]
             ])

        self.g_ = self.rotation_Y @ self.g_

    def rotate_z(self, theta):
        self.rotation_Z = np.array(
            [[np.cos(theta), -np.sin(theta), 0., 0.],
             [np.sin(theta), np.cos(theta), 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]
             ])

        self.g_ = self.rotation_Z @ self.g_

    def translate(self, p, q, r):
        self.translation_matrix = np.array([[1., 0., 0, p],
                                            [0., 1., 0, q],
                                            [0., 0., 1, r],
                                            [0., 0., 0., 1.]
                                            ])

        self.g_ = self.translation_matrix @ self.g_
        print(self.g_.shape)

    def trans_rigide(self, theta=0, omega=0, phi=0, p=0, q=0, r=0):
        self.rotate_x(theta)
        self.rotate_y(omega)
        self.rotate_z(phi)
        self.translate(p, q, r)

    def similitude(self, d=1, theta=0, omega=0, phi=0, p=0, q=0, r=0):
        self.rotate_x(theta)
        self.rotate_y(omega)
        self.rotate_z(phi)

        # Scale
        self.g_ *= d

        # Then translate
        self.translate(p, q, r)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("param", nargs='+', help='Parameters')
    parser.add_argument("--transformation", type=str,
                        help='rigide/similitude', default='rigide')
    args = parser.parse_args()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    grid = Grid3D()

    ax.scatter(grid.g_[0, :], grid.g_[1, :], grid.g_[2, :], label='raw')

    if args.transformation == 'rigide':
        grid.trans_rigide(theta=float(args.param[1]),
                          omega=float(args.param[2]),
                          phi=float(args.param[3]),
                          p=float(args.param[4]),
                          q=float(args.param[5]),
                          r=float(args.param[6]))

        ax.scatter(grid.g_[0, :], grid.g_[1, :], grid.g_[2, :], label='rigid')

    elif args.transformation == 'similitude':
        grid.similitude(d=float(args.param[0]),
                        theta=float(args.param[1]),
                        omega=float(args.param[2]),
                        phi=float(args.param[3]),
                        p=float(args.param[4]),
                        q=float(args.param[5]),
                        r=float(args.param[6]))

        ax.scatter(grid.g_[0, :], grid.g_[1, :], grid.g_[2, :], label='similitude')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.legend()

    plt.savefig(f'transformation_{args.transformation}')
    plt.show()

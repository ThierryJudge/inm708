import numpy as np
from matplotlib import pyplot as plt


class Grid3D:

    def __init__(self):
        """
        Création de la grille

        La grille a 4 dimensions et la première représente les coordonées
        plus un 1.
        in[1]:  g_[:, 7, 14, 3]
        Out[1]: array([7., 14., 3., 1.])
        """

        self.g_ = np.mgrid[-10:10, -10:10, 0:5]
        self.g_ = self.g_.reshape((3, -1)).T
        self.g_ = np.concatenate([self.g_, np.ones(len(self.g_))[:, None]],
                                 axis=1)

        # self.g_ = np.mgrid[0:20, 0:20, 0:5]
        # self.g_ = np.concatenate((self.g_,
        #                          np.ones((1,
        #                                   self.g_.shape[1],
        #                                   self.g_.shape[2],
        #                                   self.g_.shape[3]))))
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

        self.g_ = self.g_ @ self.rotation_X

    def rotate_y(self, theta):
        self.rotation_Y = np.array(
            [[np.cos(theta), 0., np.sin(theta), 0.],
             [0., 1., 0., 0],
             [-np.sin(theta), 0, np.cos(theta), 0.],
             [0., 0., 0., 1.]
             ])

        self.g_ = self.g_ @ self.rotation_Y

    def rotate_z(self, theta):
        self.rotation_Z = np.array(
            [[np.cos(theta), -np.sin(theta), 0., 0.],
             [np.sin(theta), np.cos(theta), 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]
             ])

        self.g_ = self.g_ @ self.rotation_Z

    def translate(self, p, q, r):
        self.translation_matrix = np.array([[1., 0., 0, p],
                                            [0., 1., 0, q],
                                            [0., 0., 1, r],
                                            [0., 0., 0., 1.]
                                            ])

        self.g_ = self.g_ @ self.translation_matrix

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
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    grid = Grid3D()

    ax.scatter(grid.g_[:, 0], grid.g_[:, 1], grid.g_[:, 2], label='raw')

    grid.trans_rigide(theta=0,
                      omega=0,
                      phi=45,
                      p=0,
                      q=0,
                      r=0)

    ax.scatter(grid.g_[:, 0], grid.g_[:, 1], grid.g_[:, 2], label='transformed')

    grid.__init__()
    grid.similitude(d=0.75,
                    theta=0,
                    omega=0,
                    phi=45,
                    p=0,
                    q=0,
                    r=0)

    ax.scatter(grid.g_[:, 0], grid.g_[:, 1], grid.g_[:, 2], label='similituted')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

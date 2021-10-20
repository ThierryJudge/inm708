from collections.abc import Sequence
from pathlib import Path
from typing import Tuple, List

import imageio
from scipy.ndimage import affine_transform
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


class Optimizer:
    def __init__(self, eps: float):
        self.eps = eps
        self.variable_history = None

    def step(self, variables: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.variable_history is None:
            self.variable_history = variables[None]
        else:
            self.variable_history = np.concatenate([self.variable_history, variables[None]], axis=0)

    def plot_variables(self, show=False, names: List = None):
        plt.figure()
        plt.plot(self.variable_history)
        if names:
            plt.legend(names)
        if show:
            plt.show()


class GradientDescent(Optimizer):
    def __init__(self, eps: float, momentum: float = 0):
        super().__init__(eps)
        self.momentum = momentum
        self.v = None

    def step(self, variables: np.ndarray, grads: np.ndarray) -> np.ndarray:
        super(GradientDescent, self).step(variables, grads)

        if self.v is None:
            self.v = np.zeros(len(variables))

        self.v = self.momentum * self.v + self.eps * grads
        variables = variables - self.v

        return variables

    def to_string(self):
        return f"{self.__class__.__name__}-eps={self.eps}-mom={self.momentum}"


class Adam(Optimizer):
    def __init__(self, eps, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(eps)
        self.m, self.v = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 1

    def step(self, variables: np.ndarray, grads: np.ndarray) -> np.ndarray:
        super(Adam, self).step(variables, grads)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_corr = self.m / (1 - self.beta1 ** self.t)
        v_corr = self.v / (1 - self.beta2 ** self.t)

        variables = variables - self.eps * (m_corr / (np.sqrt(v_corr) + self.epsilon))

        self.t += 1

        return variables

    def to_string(self):
        return f"{self.__class__.__name__}-eps={self.eps}"

import numpy as np
from math_packages import NUMPY, TORCH, mathpack_convert
from vector_fields import vf_dubins_car_phaseonly
from plotting import plot_flow_samples
from box_range import TUNCALI_X0


"""
car.py
------
This file defines controllers for a Dubins car which receive angle and distance differences from the target path as
inputs. The main object defined here is a simple feedforward neural network used in the 
`Tuncali et al. paper <https://arxiv.org/abs/1804.03973>`_.

When executed as the main module, the phase portraits of selected controllers are plotted.
"""


def no_control(x, mp=NUMPY):
    return 0


def sample_control(x, mp=NUMPY):
    return - x[0] - x[1]


def sample_control2(x, mp=NUMPY):
    return - 0.1 * x[0] - x[1]


def sample_control3(x, mp=NUMPY):
    return - 2 * x[0] - 2 * x[1]


class TuncaliFeedforwardController:
    """A feedforward neural network used to control a Dubins car.
    Based on the paper `Reasoning about Safety of Learning-Enabled Components in Autonomous Cyber-physical Systems
    <https://arxiv.org/abs/1804.03973>`_."""

    def __init__(self, weights=None, n_hidden=3):
        if weights is None:
            weights = np.random.rand(3 * n_hidden)
        else:
            weights = np.array(weights)

        if weights.size != 3 * n_hidden:
            raise ValueError("bad tuncali controller init")
        weights = weights.flatten()
        weights = weights.astype(float)
        first_layer, self.second_layer = weights[:2 * n_hidden], weights[2 * n_hidden:]
        self.first_layer = first_layer.reshape(n_hidden, 2)
        self.second_layer = self.second_layer.reshape(1, n_hidden)
        self.n_hidden = n_hidden

    def _prep_for_package(self, mp):
        self.first_layer = mathpack_convert(self.first_layer, mp)
        self.second_layer = mathpack_convert(self.second_layer, mp)
        if mp == TORCH:
            self.first_layer.requires_grad = True
            self.second_layer.requires_grad = True

    def __call__(self, x, mp=NUMPY):
        """Return the value of the control signal at the given point."""
        x = mathpack_convert(x, mp)
        if mp.size(x) != 2:
            raise ValueError("bad tuncali controller eval")
        self._prep_for_package(mp)

        ret = mp.matmul(self.first_layer, mp.v2c(x))
        ret = mp.tanh(ret)
        ret = mp.matmul(self.second_layer, ret)
        ret = mp.tanh(ret)
        return -ret[0][0]


if __name__ == '__main__':
    starting_points = TUNCALI_X0.samples(count=1000)

    # Found with grad descent
    init_weights = [0.4163, 4.0553, 0.3847, 3.7471, 0.3620, 3.3097, 4.1804, 3.8586, 3.3788]
    tc = TuncaliFeedforwardController(init_weights, n_hidden=3)

    # Found with grad descent
    init_weights = [2.0444, 0.0249, 1.8784, 0.3100, 2.2064, 0.1886, 2.3436, 1.6884, 2.3863]
    tc2 = TuncaliFeedforwardController(init_weights, n_hidden=3)

    # Found with CMAES
    init_weights = [1.6570, 2.9650, 1.9282, 1.2390, 1.0530, 0.7120, 2.0271, 2.1247, 1.6462]
    tc3 = TuncaliFeedforwardController(init_weights, n_hidden=3)

    for idx, f in enumerate([no_control, sample_control, sample_control2, sample_control3, tc, tc2, tc3]):
        vf = vf_dubins_car_phaseonly(f)
        if idx == 0:
            plot_flow_samples(starting_points, vf, x_dim=0, y_dim=1, times=np.arange(0, 10, 1E-4), title='base paths')
        else:
            plot_flow_samples(starting_points, vf, x_dim=0, y_dim=2, times=np.arange(0, 10, 1E-4),
                              xlabel='d_err', ylabel='th_err', title='sample control ' + str(idx + 1))

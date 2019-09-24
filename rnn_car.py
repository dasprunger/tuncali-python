import torch
import math
import numpy as np
from math_packages import TORCH, NUMPY, mathpack_convert
from box_range import RECURRENT_TRAIN, RECURRENT_TEST
from plotting import plot_flow_samples
from vector_fields import vf_int_dubins


"""
rnn_car.py
----------
This file defines a recurrent Dubins car controller which also has access to extra state, namely the integral of the
distance error. This gives it some ability to cope with curved tracks, but we haven't come up with a proof of safety.
"""


# First, some simple controllers for benchmarking
def pi_ctl(x, mp=NUMPY):
    return - x[0] - x[1] - 0.2 * x[2]


def p_ctl(x, mp=NUMPY):
    return - x[0] - x[1]


# Some track curvature functions
def c_track(g):
    return 0.5 if g < np.pi else -0.5


def c_track2(g):
    return (-1) ** math.floor(g/np.pi) * 0.5


def straight_track(g):
    return 0


class RecurrentController:
    """ This recurrent controller expects inputs in the format (d_err, th_err, \\int d_err). """
    def __init__(self, weights=None, n_hidden=3):
        if weights is None:
            weights = torch.rand(4 * n_hidden)
        else:
            weights = mathpack_convert(weights, TORCH)
            weights = torch.flatten(weights)

        if weights.numel() != 4 * n_hidden:
            raise ValueError("bad recurrent controller init")

        self.first_layer = weights[:3 * n_hidden].reshape(n_hidden, 3)
        self.first_layer.requires_grad = True
        self.second_layer = weights[3 * n_hidden:].reshape(1, n_hidden)
        self.second_layer.requires_grad = True
        self.n_hidden = n_hidden

    def _prep_for_package(self, mp):
        self.first_layer = mathpack_convert(self.first_layer, mp)
        self.second_layer = mathpack_convert(self.second_layer, mp)
        if mp == TORCH:
            self.first_layer.requires_grad = True
            self.second_layer.requires_grad = True

    def __call__(self, x, mp=TORCH):
        x = mathpack_convert(x, mp)
        if mp.size(x) != 3:
            raise ValueError("bad recurrent controller eval")
        self._prep_for_package(mp)

        ret = mp.matmul(self.first_layer, mp.v2c(x))
        ret = mp.tanh(ret)
        ret = mp.matmul(self.second_layer, ret)
        ret = mp.tanh(ret)
        return -ret[0][0]


def run(controller, initial_point, rounds=3, step_size=1E-3):
    zlist = [initial_point]
    curve = 1.6 * torch.rand(1) - 0.8
    while rounds > 0:
        # we linearize the differential equation at the point obtained in the last round
        dd = torch.sin(zlist[-1][1]).reshape(1)
        dth = controller(zlist[-1]).reshape(1) + curve
        dintd = zlist[-1][0].reshape(1)

        # step
        zlist.append(zlist[-1] + step_size * torch.cat((dd, dth, dintd), 0))
        rounds -= 1
    return zlist


def train_new(steps=2E4):
    model = RecurrentController()
    points = RECURRENT_TRAIN.samples(100, mp=TORCH)

    while steps > 0:
        for point in points:
            res = run(model, point, rounds=5)
            cost = 5E2 * res[-1][0] ** 2 + res[-1][1] ** 2
            cost.backward()

        with torch.no_grad():
            model.first_layer -= model.first_layer.grad
            model.second_layer -= model.second_layer.grad
            model.first_layer.grad.zero_()
            model.second_layer.grad.zero_()

        steps -= 1

        if steps % 1000 == 0:
            print("With {} steps to go:".format(steps))
            print(model.first_layer)
            print(model.second_layer)


if __name__ == '__main__':
    # train_new()

    ## This RNN was trained with J = 1E2 * d_err^2 + th_err^2 for 20k steps on 30 randomized points
    # weights = [0.2170, 2.5536, -0.0069, 0.3962, 5.4161, -0.0538, 0.2441, 3.2533, -0.0087, 2.6079, 5.5339, 3.0841]
    # rnn = RecurrentControllerTorch(weights)
    # plot_samples(rnn)

    ## This RNN was trained with J = 5E2 * d_err^2 + th_err^2 for 20k steps on 30 randomized points
    # weights = [1.5698,  2.8923,  0.0867, 1.2882,  1.9319, -0.0192, 1.6073,  3.3255,  0.1102, 3.1529, 2.3237, 3.8094]
    # rnn = RecurrentControllerTorch(weights)
    # plot_samples(rnn)

    ## This RNN was trained with J = 5E2 * d_err^2 + th_err^2 for 20k steps on 100 randomized points
    # weights = [1.8984e+00, 3.3224e+00, 2.2690e-03,
    #            4.3399e-01, 9.2616e-01, 2.5299e-01,
    #            2.3175e+00, 4.5838e+00, 1.3214e-02,
    #            4.0371, 0.9050, 5.2937]
    # rnn = RecurrentControllerTorch(weights)
    # plot_samples(rnn)

    ## This RNN was trained with J = 5E2 * d_err^2 + th_err^2 for 20k steps on 100 randomized points
    # with randomized curvatures lower than 0.8v in abs value
    # weights = [3.4672, 4.0470, 0.3985, 2.9772, 2.8131, 0.3126, 1.7862, 2.0513, 0.5851, 5.4433, 4.2083, 2.7482]
    # rnn = RecurrentControllerTorch(weights)
    # plot_samples(rnn, 'Stateful (\\int d) network on straight tracks, sampled widely', wide=True)

    ## This RNN was trained with J = 5E2 * d_err^2 + th_err^2 for 20k steps on 100 randomized points, uncorrelating d and intd
    # with randomized curvatures lower than 0.8v in abs value
    weights = [2.4516,  5.1544, -0.1221, 2.4652,  0.9492,  0.6990, 1.3925,  2.9339, -0.0900, 5.6492, 1.4950, 3.1075]
    rnn = RecurrentController(weights)
    r = torch.rand(1)
    vf = vf_int_dubins(rnn, curve=lambda y: r)
    samples = RECURRENT_TEST.samples(300, mp=NUMPY)
    plot_flow_samples(samples, vf, x_dim=0, y_dim=1, times=np.arange(0, 10, 1E-4),
                      title='Stateful (\\int d) network on curved tracks')

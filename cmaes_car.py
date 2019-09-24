import numpy as np
import cma
from car import TuncaliFeedforwardController
from vector_fields import vf_dubins_car_phaseonly, flow
from box_range import TUNCALI_U

"""
cmaes_car.py
------------
This file implements training of controllers using CMA-ES, as was done in the original Tuncali et al. paper.

See nn_car.py for a gradient descent style optimization for the same kinds of controllers.
"""


def cost(run, tc):
    """ This is the cost function in the Tuncali paper exactly. """
    ctl_cost = 100 * sum([tc([_[0], _[2]]) ** 2 for _ in run])
    x_cost = 100 * sum(run[:, 0] * run[:, 0])
    angle_cost = 1E5 * sum(run[:, 2] * run[:, 2])
    final_cost = 1E3 * (run[-1, 0]**2 + (1 - run[-1, 1])**2)
    return x_cost + angle_cost + final_cost + ctl_cost


def cost2(run, tc):
    ctl_cost = sum([tc([_[0], _[2]]) ** 2 for _ in run])
    x_cost = 1000 * sum(run[:, 0] * run[:, 0])
    angle_cost = 1000 * sum(run[:, 2] * run[:, 2])
    return ctl_cost + x_cost + angle_cost


def cost3(run, tc):
    x_cost = 1000 * run[-1, 0] * run[-1, 0]
    angle_cost = 10 * run[-1, 2] * run[-1, 2]
    return x_cost + angle_cost


def cost4(run, tc):
    x_cost = 1000 * run[-1, 0] * run[-1, 0]
    angle_cost = 1 * run[-1, 2] * run[-1, 2]
    return x_cost + angle_cost


def cost5(run, tc):
    x_cost = 1000 * run[-1, 0] * run[-1, 0]
    angle_cost = 1 * run[-1, 2] * run[-1, 2]
    ctl_cost = tc([run[-1, 0], run[-1, 2]]) ** 2
    return x_cost + angle_cost + ctl_cost


def cost6(run, tc):
    """ This is the cost function from the Tuncali paper, with angle and distance weights swapped. """
    ctl_cost = 1E2 * sum([tc([_[0], _[2]]) ** 2 for _ in run])
    x_cost = 1E5 * sum(run[:, 0] * run[:, 0])
    angle_cost = 1E2 * sum(run[:, 2] * run[:, 2])
    final_cost = 1E3 * (run[-1, 0]**2 + (1 - run[-1, 1])**2)
    return x_cost + angle_cost + final_cost + ctl_cost


def to_opt(weights, cost_function, n_hidden=3):
    samples = TUNCALI_U.samples()
    tc = TuncaliFeedforwardController(weights, n_hidden=n_hidden)
    ret = 0
    for pt in samples:
        vf = vf_dubins_car_phaseonly(tc)
        run = flow(pt, vf, control_points=np.arange(0, 1, 1E-4))
        ret += cost_function(run, tc)
    return ret


def train(cost_function, n_hidden=3):
    # weights = np.random.rand(3 * n_hidden)
    weights = np.ones(9)
    x, es = cma.fmin2(lambda _: to_opt(_, cost_function), weights, 1, options={'maxiter': 50, 'popsize': 152})
    print(x, es)


if __name__ == '__main__':
    train(cost6)

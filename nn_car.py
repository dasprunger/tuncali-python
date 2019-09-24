import torch
from box_range import TUNCALI_PHASE_X0
from car import TuncaliFeedforwardController
from math_packages import TORCH


"""
nn_car.py
---------
This file implements basic gradient descent training for feedforward neural network controllers 
(TuncaliFeedforwardController). It should be thought of as a sibling to cmaes_car.py, just using a different 
optimization technique.
"""


def run(controller, initial_point, rounds=3, step_size=1E-3):
    zlist = [initial_point]
    while rounds > 0:
        dd = torch.sin(zlist[-1][1]).reshape(1)
        dth = controller(zlist[-1], mp=TORCH).reshape(1)  # call the controller with TORCH to maintain gradients
        zlist.append(zlist[-1] + step_size * torch.cat((dd, dth), 0))
        rounds -= 1
    return zlist


def cost(res):
    return 5E2 * res[-1][0] ** 2 + res[-1][1] ** 2


def train(controller, cost, steps=2E4, resample_every=100):
    samples = TUNCALI_PHASE_X0.samples(count=30, mp=TORCH)
    while steps > 0:
        for point in samples:
            res = run(controller, point)
            c = cost(res)
            c.backward()

        with torch.no_grad():
            controller.first_layer -= controller.first_layer.grad
            controller.second_layer -= controller.second_layer.grad
            controller.first_layer.grad.zero_()
            controller.second_layer.grad.zero_()

        steps -= 1

        if steps % 1000 == 0:
            print("With {} steps to go:".format(steps))
            print(controller.first_layer)
            print(controller.second_layer)

        if resample_every and steps % resample_every == 0:
            samples = TUNCALI_PHASE_X0.samples(count=30, mp=TORCH)

    return controller


if __name__ == '__main__':
    ctl = TuncaliFeedforwardController()
    train(ctl, cost)



import numpy as np
import matplotlib.pyplot as plt
from vector_fields import flow, vf_dubins_car_phaseonly
from math import pi
from box_range import TUNCALI_PHASE_X0
from barrier import Form


"""
plotting.py
-----------
Some basic plotting facilities to help visualizing flows and barriers.

This module isn't very well structured yet---I don't know what is generally useful here.
"""


def plot_flow_samples(initial_points, vector_field, x_dim=-1, y_dim=0, times=None, title='asdfadsf',
                      xlabel='x-axis', ylabel='y-axis', aspect=1):
    fig = plt.figure(1, figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set(aspect=aspect, title=title, xlabel=xlabel, ylabel=ylabel)
    if times is None:
        times = np.arange(0, 10, 1E-4)

    for pt in initial_points:
        sln = flow(pt, vector_field, control_points=times)
        x_series = times if x_dim < 0 else sln[:, x_dim]
        y_series = times if y_dim < 0 else sln[:, y_dim]
        ax.plot(x_series, y_series)

    plt.tight_layout()
    plt.show()


def plot_vector_field():
    X = np.arange(-2, 2, 0.2)
    Y = np.arange(-pi/2, pi/2, pi/20)
    X, Y = np.meshgrid(X, Y)
    U = np.sin(Y)
    V = - 0.1 * X - Y

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set(aspect=1)
    ax.set_title('vector field for controller 2')
    q = ax.quiver(X, Y, U, V, angles='xy', width=0.002)

    pts = TUNCALI_PHASE_X0.samples(300)
    for pt in pts:
        sln = flow(pt, lambda x, t: np.array([np.sin(x[1]), -0.1 * x[0] - x[1]]), control_points=np.arange(0, 10, 1E-4))
        ax.plot(sln[:, 0], sln[:, 1])

    barrier = Form(2, values=np.array([1.5840647673705432, 2 * 0.642477059258331, 2.470329836702939]))

    for l in [2.1]:
        barrier.plot_level_set(l, fas=(fig, ax, False))
        pos_sln = barrier.polar_ellipsoidal_series(l)
        X, Y = pos_sln[:, 0], pos_sln[:, 1]
        U = np.sin(Y)
        V = - 0.1 * X - Y
        ax.quiver(X, Y, U, V, angles='xy', width=0.003, color='b')

    ax.plot([-1, -1, 1, 1, -1], [-pi/16, pi/16, pi/16, -pi/16, -pi/16], color='g')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_vector_field()


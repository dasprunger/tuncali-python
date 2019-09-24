from math_packages import NUMPY, mathpack_convert
import numpy as np
from scipy.integrate import odeint


"""
vector_fields.py
----------------
This defines several vector fields defining the behaviour of Dubins cars. 
It also provides a facility for finding the flow along a vector field.
"""


# Dubins car has the following dynamics:
#  x' = V sin theta
#  y' = V cos theta
#  theta' = u
# where u is the turn rate control, bounded by a constant in absolute value
def vf_dubins_car(control_function, v=1, curve=lambda y: 0, mp=NUMPY):
    def ret(args, t=None, mp=mp):
        x, y, theta = args[:3]
        return mathpack_convert([v * mp.sin(theta), v * mp.cos(theta), control_function(args, mp) + curve(y)], mp)
    return ret


def vf_dubins_car_phaseonly(control_function, v=1, curve=lambda y: 0, mp=NUMPY):
    def ret(args, t=None, mp=mp):
        if len(args) == 2:
            x, theta = args[:2]
            u = control_function(mp.vect_cat(x, theta), mp=mp)
            return mathpack_convert([v * mp.sin(theta), u], mp)
        else:
            x, y, theta = args[:3]
            u = control_function(mp.vect_cat(x, theta), mp=mp)
            return mathpack_convert([v * mp.sin(theta), v * mp.cos(theta), u + curve(y)], mp)
    return ret


def vf_int_dubins(control_function, v=1, curve=lambda y: 0, mp=NUMPY):
    def ret(args, t=None, mp=mp):
        if len(args) == 3:
            x, th, intx = args[:3]
            u = control_function(mp.vect_cat(x, th, intx), mp=mp)
            return mathpack_convert([v * mp.sin(th), u + curve(0), x], mp)
        else:
            x, y, th, intx = args[:4]
            u = control_function(mp.vect_cat(x, th, intx), mp=mp)
            return mathpack_convert([v * mp.sin(th), v * mp.cos(th), u + curve(y), x], mp)
    return ret


def flow(ip, vf, control_points=None):
    """Return a time series representing the solution to the differential equation represented by the vector field with
    the starting point. Uses numpy.

    :param ip: initial point of the differential equation
    :param vf: vector field defining the differential equation
    :param control_points: optional numpy array giving the times where the solution should be given.
    Defaults to all multiples of 10^(-4) in [0, 1] interval."""
    if control_points is None:
        control_points = np.arange(0, 1, 1E-4)

    control_points = mathpack_convert(control_points, NUMPY)
    ip = mathpack_convert(ip, NUMPY)

    def safe_vf(x, t):
        return mathpack_convert(vf(x, t), NUMPY)

    return odeint(safe_vf, ip, control_points)

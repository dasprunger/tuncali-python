import numpy as np
import torch
from scipy.optimize import linprog
from dreal import Variable, And, Not, CheckSatisfiability, sin, cos
from rnn_car import RecurrentController
from barrier import Form, qf_monomial_count, dim_from_monomial_count, square_coord_to_linear
from math_packages import NUMPY, DREAL, mathpack_convert
from box_range import TUNCALI_PHASE_X0, TUNCALI_PHASE_U, TUNCALI_X0, TUNCALI_U
from car import TuncaliFeedforwardController, sample_control, sample_control2
from vector_fields import vf_dubins_car_phaseonly
from itertools import product

"""
constraint_generator
"""

EPSILON = 1E-6
SOLUTION_TOLERANCE = 1 * EPSILON


def safe_region_samples(count, region=TUNCALI_PHASE_U, exclude=TUNCALI_PHASE_X0, mp=NUMPY):
    samples = region.samples(count, mp=mp)
    if exclude is not None:
        samples = [_ for _ in samples if _ not in exclude]

    while len(samples) < count:
        new_samples = region.samples(count - len(samples), mp=mp)
        if exclude is not None:
            new_samples = [_ for _ in new_samples if _ not in exclude]
        samples.extend(new_samples)
    return mathpack_convert(samples, mp)


def generate_barrier_candidate(matrix, vector):
    solve = linprog(np.zeros(matrix.shape[1]), matrix, vector, method='interior-point')
    p = Form(dim_from_monomial_count(matrix.shape[1]), values=solve['x'])
    return p


def dReal_check(barrier, vector_field, region=TUNCALI_PHASE_U, exclude=TUNCALI_PHASE_X0):
    final_conjuncts = [region.formula(delta=EPSILON, strict=True), Not(exclude.formula(strict=True))]

    # Check that there are feasible points in this region
    sanity = CheckSatisfiability(And(*final_conjuncts), EPSILON)
    if sanity is None:
        raise RuntimeError("empty satisfiability region")

    # ... and that having negative Lie derivative is required.
    ld = barrier.lie_derivative(vector_field, region.variables, mp=DREAL)
    final_conjuncts.append(ld > - EPSILON)
    result = CheckSatisfiability(And(*final_conjuncts), EPSILON)
    return result  # should be None


def find_barrier_for_vector_field(vf):
    samples = safe_region_samples(300)
    matrix, vector = Form.constraints(vf, samples), np.array([-SOLUTION_TOLERANCE] * len(samples))
    p = generate_barrier_candidate(matrix, vector)

    check = dReal_check(p, vf)
    TRIES = 100
    VARS = TUNCALI_PHASE_U.variables
    while check is not None and TRIES > 0:
        if TRIES % 10 == 0:
            print(check)
        dim = len(VARS)
        new_pts = list(product(*[[check[_].lb(), check[_].ub()] for _ in range(dim)]))
        new_pts = np.array(new_pts)
        new_matrix, new_vector = Form.constraints(vf, new_pts), np.array([-SOLUTION_TOLERANCE] * len(new_pts))
        matrix = np.append(matrix, new_matrix, axis=0)
        vector = np.append(vector, new_vector, axis=0)
        print('trying a {} row constraint...'.format(matrix.shape[0]))

        p = generate_barrier_candidate(matrix, vector)
        check = dReal_check(p, vf)
        TRIES -= 1

    if check is None:
        print('success!')
        print(str(p))
        p.plot_level_set(1)
    else:
        print('failure!')


if __name__ == '__main__':
    init_weights = [0.4163, 4.0553, 0.3847, 3.7471, 0.3620, 3.3097, 4.1804, 3.8586, 3.3788]
    tc = TuncaliFeedforwardController(init_weights, n_hidden=3)
    vf = vf_dubins_car_phaseonly(tc)

    find_barrier_for_vector_field(vf)


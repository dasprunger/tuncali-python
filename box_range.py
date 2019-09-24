from math_packages import NUMPY, TORCH, mathpack_convert
import dreal as dr
from math import pi


"""
box_range.py
------------
This file provides axis-aligned subsets of R^k, including a sampling facility and a way to get dReal formulas.
"""


class AxisAlignedRange:
    def __init__(self, bounds, variables=None):
        self.bounds = bounds
        self.widths = [_[1] - _[0] for _ in bounds]
        if variables:
            self.variables = variables
        else:
            self.variables = [dr.Variable('var' + str(i)) for i in range(len(bounds))]

    def __contains__(self, item):
        if len(item) != len(self.bounds):
            return False
        return all([self.bounds[i][0] <= item[i] <= self.bounds[i][1] for i in range(len(self.bounds))])

    def samples(self, count=10, mp=NUMPY):
        """Return :count: samples from """
        if mp not in [NUMPY, TORCH]:
            raise RuntimeError("Cannot sample without numpy-derived package.")
        samples = mathpack_convert(self.widths, mp) * mp.random(count, len(self.bounds))
        samples += mathpack_convert([_[0] for _ in self.bounds], mp)
        return samples

    def formula(self, delta=0, strict=False):
        """Return a dReal formula expressing membership in this region."""
        conjuncts = []
        delta = abs(delta)
        for bound, var in zip(self.bounds, self.variables):
            if strict:
                conjuncts.append(bound[0] + delta < var)
                conjuncts.append(var < bound[1] - delta)
            else:
                conjuncts.append(bound[0] + delta <= var)
                conjuncts.append(var <= bound[1] - delta)
        return dr.And(*conjuncts)


DREAL_X, DREAL_Y, DREAL_T, DREAL_XINT = dr.Variable('x'), dr.Variable('y'), dr.Variable('t'), dr.Variable('xint')
TUNCALI_PHASE_X0 = AxisAlignedRange([[-1, 1], [-pi/16, pi/16]], variables=[DREAL_X, DREAL_T])
TUNCALI_PHASE_U = AxisAlignedRange([[-5, 5], [-pi/2, pi/2]], variables=[DREAL_X, DREAL_T])
TUNCALI_X0 = AxisAlignedRange([[-1, 1], [0, 0], [-pi/16, pi/16]], variables=[DREAL_X, DREAL_Y, DREAL_T])
TUNCALI_U = AxisAlignedRange([[-5, 5], [0, 0], [-pi/2, pi/2]], variables=[DREAL_X, DREAL_Y, DREAL_T])

RECURRENT_TRAIN = AxisAlignedRange([[-5, 5], [-pi / 2, pi / 2], [-10, 10]], variables=[DREAL_X, DREAL_T, DREAL_XINT])
RECURRENT_TEST = AxisAlignedRange([[-2, 2], [-pi / 2, pi / 2], [0, 0]], variables=[DREAL_X, DREAL_T, DREAL_XINT])

if __name__ == '__main__':
    box = TUNCALI_X0
    print(box.samples())
    print([0, 0.1] in box)
    print(box.formula())

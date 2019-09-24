from functools import reduce
from copy import copy
import dreal as dr
import torch
import numpy as np


"""
math_packages.py
----------------
A big annoyance in this work is needing various capabilities of different scientific computing packages: we use numpy
for plotting and (numerically) solving differential equations, torch for automatic differentiation and gradient descent,
and dReal for satisfiability checks. These all have their own implementations of various functions (tanh, sin, cos) that
cannot be used interchangeably. Moreover, some packages are missing features available in others, notably many of the
tensor-related operations like matrix multiplication aren't present in dReal.

A MathPackage is an abstraction wrapping functions of these libraries in a single interface. We don't aim for
comprehensiveness; we're only interested in the functionality needed for this project. We mostly follow numpy, and most
of the work in this file implements extensions of dReal.

We also include a function mathpack_convert which takes data (vanilla Python lists for dReal, torch.tensors and
numpy.ndarrays) in the various formats and restructures it into a format usable by the other packages. 
"""


class MathPackage:
    """A wrapper for a math library to provide a unified interface."""
    FUNCTION_NAMES = ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'matmul', 'random', 'shape', 'size', 'zeros',
                      'v2r', 'v2c', 'transpose', 'vect_cat']

    def __init__(self, d):
        for name in d.keys():
            self.register_function(name, d[name])

    def register_function(self, name, implementation):
        if name in MathPackage.FUNCTION_NAMES:
            setattr(self, name, implementation)


def torch_vector_cat(*args):
    c = [_.reshape(_.numel()) for _ in args]
    return torch.cat(tuple(c))


def np_vector_cat(*args):
    c = [_.reshape(_.size) for _ in args]
    return np.concatenate(tuple(c))


TORCH = MathPackage({'sin': torch.sin, 'cos': torch.cos, 'tan': torch.tan, 'tanh': torch.tanh,
                     'matmul': torch.matmul, 'random': torch.rand, 'transpose': lambda x: x.t(),
                     'shape': lambda x: tuple(x.size()), 'size': lambda x: x.numel(), 'zeros': torch.zeros,
                     'v2c': lambda x: x.reshape(x.numel(), 1), 'v2r': lambda x: x.reshape(1, x.numel()),
                     'vect_cat': torch_vector_cat
                     })
NUMPY = MathPackage({'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'tanh': np.tanh,
                     'matmul': np.matmul, 'random': np.random.rand, 'transpose': lambda x: x.transpose(),
                     'shape': lambda x: x.shape, 'size': lambda x: x.size, 'zeros': np.zeros,
                     'v2c': lambda x: x.reshape(x.size, 1), 'v2r': lambda x: x.reshape(1, x.size),
                     'vect_cat': np_vector_cat,
                     })


def simple_shape(mat):
    ret = []
    while hasattr(mat, '__len__'):
        ret.append(len(mat))
        mat = mat[0]
    return ret


def simple_size(mat):
    shape = simple_shape(mat)
    return reduce(int.__mul__, shape, 1)


def simple_zero(dims):
    ret = 0
    for dim in dims[::-1]:
        ret = [copy(ret) for _ in range(dim)]
    return ret


def simple_matmul(mat1, mat2, math_package=None):
    mp = math_package
    s1, s2 = mp.shape(mat1), mp.shape(mat2)
    if len(s1) != 2 or len(s2) != 2 or s1[1] != s2[0]:
        raise RuntimeError("mismatched matrix dimensions in matrix multiply: {!s} and {!s}".format(s1, s2))
    ret = mp.zeros((s1[0], s2[1]))
    for i in range(s1[0]):
        for j in range(s2[1]):
            for k in range(s1[1]):
                ret[i][j] += mat1[i][k] * mat2[k][j]
    return ret


def simple_transpose(mat, math_package=None):
    mp = math_package
    s = mp.shape(mat)
    if len(s) < 2:
        return mat
    if len(s) > 2:
        raise RuntimeError("cannot transpose dimension >2 tensor")
    ret = mp.zero(s[1], s[0])
    for i in range(s[1]):
        for j in range(s[0]):
            ret[i][j] = mat[j][i]
    return ret


def simple_column_vector_embedding(vect):
    return [[_] for _ in vect]


def simple_row_vector_embedding(vect):
    return [vect]


def broadcast(fn, data):
    shape = simple_shape(data)
    if len(shape) == 1:
        return [fn(_) for _ in data]
    else:
        return [broadcast(fn, _) for _ in data]


def repeated_extend(*lists):
    ret = []
    for l in lists:
        if hasattr(l, '__iter__'):
            ret.extend(l)
        else:
            ret.append(l)
    return ret


DREAL = MathPackage({'sin': dr.sin, 'cos': dr.cos, 'tan': dr.tan, 'tanh': lambda x: broadcast(dr.tanh, x),
                     'random': lambda x: np.random.rand(x).tolist(),
                     'shape': simple_shape, 'size': simple_size, 'zeros': simple_zero,
                     'v2c': simple_column_vector_embedding, 'v2r': simple_row_vector_embedding,
                     'vect_cat': repeated_extend,
                     })

DREAL.register_function('matmul', lambda *x: simple_matmul(*x, math_package=DREAL))
DREAL.register_function('transpose', lambda *x: simple_transpose(*x, math_package=DREAL))


def mathpack_convert(data, target_math_package):
    if isinstance(data, list):  # source package is DREAL
        if target_math_package == DREAL:
            return data
        elif target_math_package == NUMPY:
            return np.array(data)
        elif target_math_package == TORCH:
            return torch.tensor(data).float()

    elif isinstance(data, np.ndarray):  # source package is NUMPY
        if target_math_package == DREAL:
            return data.tolist()
        elif target_math_package == NUMPY:
            return data
        elif target_math_package == TORCH:
            return torch.tensor(data).float()

    elif isinstance(data, torch.Tensor):  # source package is TORCH
        if target_math_package == DREAL:
            return data.detach().numpy().tolist()
        elif target_math_package == NUMPY:
            return data.detach().numpy()
        elif target_math_package == TORCH:
            return data




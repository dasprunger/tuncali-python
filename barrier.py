import numpy as np
from math_packages import NUMPY, mathpack_convert
import matplotlib.pyplot as plt


"""
barrier.py
----------
This file implements quadratic form templates for use as barrier certificates.

Throughout, 'dimension' refers to the number of variables in the quadratic form (e.g. 2 for ax^2 + bxy + cy^2), and
'monomial count' refers to the number of monomials involved in that quadratic form (e.g. 3 for ax^2 + bxy + cy^2).
The 'square representation' of the form is the symmetric matrix often used (i.e. P in x^TPx) and coordinates refer to 
the indexes in this matrix, while the 'linear representation' is the list of monomial coefficients (e.g. [a, b, c] for 
ax^2 + bxy + cy^2).
"""


def qf_monomial_count(dim):
    """Return the monomial count given the number of variables of the form."""
    return (dim * (dim + 1)) // 2


def dim_from_monomial_count(count):
    """Return the dimension of a form given the monomial count."""
    return int(np.floor(np.sqrt(2*count)))


def square_coord_to_linear(i, j, matrix_dim):
    """Return the linear coordinate corresponding to a square coordinate.
    Inverse function is linear_coord_to_square."""
    if i < 0 or i >= matrix_dim or j < 0 or j >= matrix_dim:
        return None
    if i > j:
        i, j = j, i
    previous_entries = matrix_dim * i - ((i * (i + 1)) // 2)
    return previous_entries + j


def linear_coord_to_square(k, matrix_dim):
    """Return the square coordinate corresponding to a linear coordinate.
    Inverse function is square_coord_to_linear."""
    if k < 0 or k >= qf_monomial_count(matrix_dim):
        return None
    i = 0
    while k >= matrix_dim - i:
        k -= matrix_dim - i
        i += 1
    return i, k + i


def quadratic_solver(a, b, c):
    if b ** 2 - 4 * a * c < 0:
        return []
    elif b ** 2 - 4 * a * c == 0:
        return [-b / (2 * a)]
    else:
        branch = np.sqrt(b ** 2 - 4 * a * c) / (2 * a)
        return [-b / (2 * a) + branch, -b / (2 * a) - branch]


def ellipsoidal_series(a, b, c, l):
    pos_sln, neg_sln = [], []
    high_range = np.sqrt(l / (a - b ** 2 / (4 * c)))
    for x in np.arange(-high_range, high_range, 2 * high_range / 100):
        slns = quadratic_solver(c, b * x, a * x ** 2 - l)
        if len(slns) == 0:
            continue
        elif len(slns) == 1:
            pos_sln.append([x, slns[0]])
        else:
            pos_sln.append([x, slns[0]])
            neg_sln.append([x, slns[1]])
    pos_sln.extend(neg_sln[::-1])
    pos_sln.append(pos_sln[0])
    return np.array(pos_sln)


class Form:
    """A quadratic form with some functionality to treat it as a barrier certificate.

    This should be extended to accommodate extended forms (where the x in x^TPx may have degree > 1)."""
    def __init__(self, dimension, values=None, mp=NUMPY):
        self._monomial_count = qf_monomial_count(dimension)
        self._dimension = dimension
        self._linear_rep = mp.zeros(self._monomial_count)
        self._square_rep = mp.zeros((dimension, dimension))
        if values is not None:
            self.linear_representation = values

    def _prep_for_package(self, mp):
        self._linear_rep = mathpack_convert(self._linear_rep, mp)
        self._square_rep = mathpack_convert(self._square_rep, mp)

    @property
    def linear_representation(self):
        return self._linear_rep

    @linear_representation.setter
    def linear_representation(self, value, mp=NUMPY):
        if mp.size(value) == self._monomial_count:
            self._linear_rep = value

            dim = self._dimension
            new_square_rep = mp.zeros((dim, dim))
            for k, val in enumerate(value):
                i, j = linear_coord_to_square(k, dim)
                new_square_rep[i][j] += value[k] / 2.0
                new_square_rep[j][i] += value[k] / 2.0
            self._square_rep = new_square_rep

    @property
    def square_representation(self):
        return self._square_rep

    def lie_derivative(self, vf, pt, mp=NUMPY):
        """Return the Lie derivative of the form with respect to the vector field at the given point.

        :param vf: vector field with respect to which to take the derivative.
        :param pt: the point where the Lie derivative should be taken.
        :param mp: the math package to use to do the computation (default: numpy)."""
        self._prep_for_package(mp)
        row = mp.v2r(pt)
        col = mp.v2c(vf(pt, mp=mp))
        prod = mp.matmul(row, mp.matmul(self._square_rep, col))[0][0]
        return 2 * prod

    def __call__(self, pt, mp=NUMPY):
        """Return the value of the quadratic form at an input point."""
        self._prep_for_package(mp)
        return mp.matmul(mp.v2r(pt), mp.matmul(self._square_rep, mp.v2c(pt)))[0][0]

    @classmethod
    def constraints(cls, vf, pts, mp=NUMPY):
        """Computes the coefficients of an inequality on the linear representation of the form which must be satisfied
        in order to make the value of the Lie derivative negative at each of the given points."""
        matrix = []
        for pt in pts:
            dyn = vf(pt)
            row = [0] * qf_monomial_count(mp.size(pt))

            for i, state_value in enumerate(pt):
                for j, dynamics_value in enumerate(dyn):
                    linear_coord = square_coord_to_linear(i, j, mp.size(pt))
                    row[linear_coord] += state_value * dynamics_value
            matrix.append(row)
        return mathpack_convert(matrix, mp)

    def __str__(self):
        return str(self._square_rep)

    def polar_ellipsoidal_series(self, level):
        """Computes the (x, y)-coordinates of 100 points distributed at equal angles from the level set of the form.
        Useful for plotting."""
        ret = []
        a, b, c = self._linear_rep
        for th in np.arange(0, 2 * np.pi, 2 * np.pi / 100):
            r = np.sqrt(level / (a * np.cos(th) ** 2 + b * np.cos(th) * np.sin(th) + c * np.sin(th) ** 2))
            ret.append([r * np.cos(th), r * np.sin(th)])
        ret.append(ret[0])
        return np.array(ret)

    def plot_level_set(self, level, fas=None):
        """Plots the given level set of the quadratic form"""
        series = self.polar_ellipsoidal_series(level)

        if fas is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig, ax = fas[0], fas[1]
        # ax.set(aspect=1)
        # ax.set_title('asdf')
        ax.plot(series[:, 0], series[:, 1], color='k')

        if fas is None or fas[2]:
            plt.tight_layout()
            plt.show()

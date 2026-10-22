'''
Exponential-sum (Laplace) quadratures for 1/x on a positive interval,

    1/x ~= sum_q w_q exp(-t_q x),   x in [xmin, xmax].

The default is a trapezoidal rule after the double-exponential substitution
t = exp(u - exp(-u)) of 1/x = int_0^inf exp(-x t) dt, applied to the scaled
variable y = x / xmin in [1, R].  Both tails of the integrand decay double
exponentially, so the error falls exponentially with the number of points;
the truncation interval [u_lo, u_hi] is chosen for each (R, n) by minimizing
the maximum relative error on a logarithmic grid.
'''

import numpy


class Quadrature:
    '''Exponents t, weights w and the estimated maximum relative error of
    1/x on [xmin, xmax].'''

    def __init__(self, t, w, xmin, xmax, max_rel_error):
        self.t = t
        self.w = w
        self.xmin = xmin
        self.xmax = xmax
        self.max_rel_error = max_rel_error

    def __len__(self):
        return len(self.t)

    def __call__(self, x):
        '''sum_q w_q exp(-t_q x), the approximation to 1/x.'''
        x = numpy.asarray(x, dtype=float)
        return numpy.exp(-numpy.multiply.outer(x, self.t)) @ self.w


def _de_nodes(n, u_lo, u_hi):
    u = numpy.linspace(u_lo, u_hi, n)
    h = u[1] - u[0]
    t = numpy.exp(u - numpy.exp(-u))
    w = h * t * (1 + numpy.exp(-u))
    return t, w


def _rel_error(t, w, ratio, nsample=2000):
    y = numpy.geomspace(1.0, ratio, nsample)
    return abs((numpy.exp(-numpy.outer(y, t)) @ w) * y - 1).max()


def de_quadrature(xmin, xmax, n):
    '''Double-exponential trapezoidal exponential sum with n points.'''
    if not 0 < xmin <= xmax:
        raise ValueError('Laplace quadrature needs 0 < xmin <= xmax, got [%g, %g]'
                         % (xmin, xmax))
    if n < 2:
        raise ValueError('need at least two quadrature points')
    ratio = max(xmax / xmin, 1.0 + 1e-12)
    best = None
    for u_lo in numpy.arange(-5.0, -1.49, 0.25):
        for u_hi in numpy.arange(1.0, 5.01, 0.25):
            t, w = _de_nodes(n, u_lo, u_hi)
            err = _rel_error(t, w, ratio)
            if best is None or err < best[0]:
                best = (err, t, w)
    err, t, w = best
    return Quadrature(t / xmin, w / xmin, xmin, xmax, err)


def quadrature(xmin, xmax, n, kind='de'):
    if kind == 'de':
        return de_quadrature(xmin, xmax, n)
    raise ValueError('unknown quadrature kind %r' % kind)

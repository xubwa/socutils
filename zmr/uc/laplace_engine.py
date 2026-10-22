'''
Laplace-transformed contraction of one perturber class.

With D = lambda_k + delta_e - s_J > 0 and 1/D ~= sum_q w_q exp(-t_q D),

    a_IJ = - sum_q w_q exp(t_q (s_J - c')) sum_XY M_q[X, Y] G_q^{IJ}[X, Y]

    M_q[X, Y]      = sum_e conj(v[e, X]) v[e, Y] exp(-t_q (delta_e + c - c'))
    G_q^{IJ}[X, Y] = sum_k conj(T^I[k, X]) T^J[k, Y] exp(-t_q (lambda_k - c))

The external part M_q is built like a Laplace-MP2 intermediate and depends
only on integrals and orbital energies; the active part G_q depends only on
the active space.  c = min(lambda_k) and c' = min(s_J) keep every exponential
factor at most one.  G_q is never stored: sum_XY M_q G_q is contracted as
sum_k exp(-t_q (lambda_k - c)) T^I[k]^+ M_q T^J[k].
'''

import numpy

from socutils.zmr.lib import linalg as la
from socutils.zmr.uc import engine, laplace


class IntruderError(RuntimeError):
    pass


def denominator_range(lam, deltas, shifts):
    '''Exact [min, max] of lambda_k + delta_e - s_J.'''
    lo = lam.min() + min(d.min() for d in deltas) - shifts.max()
    hi = lam.max() + max(d.max() for d in deltas) - shifts.min()
    return lo, hi


def laplace_contract(blocks, t_amp, lam, shifts, nquad=None, kind='de', name='?', col=None):
    '''Laplace-transformed counterpart of engine.exact_contract_t.'''
    nstate = t_amp.shape[0]
    dtype = numpy.result_type(t_amp, *[v for _, v in blocks[:1]])
    a = numpy.zeros((nstate, nstate), dtype=dtype)
    shifts = numpy.asarray(shifts, dtype=float)
    dmin, dmax = denominator_range(lam, [d for d, _ in blocks], shifts)
    if dmin <= 0:
        raise IntruderError(
            'class %s: denominator range [%g, %g] is not positive; the Laplace '
            'transform does not apply (intruder state) - use method="exact"'
            % (name, dmin, dmax))
    quad = laplace.quadrature(dmin, dmax, nquad or 30, kind)

    c_act = lam.min()
    c_shift = shifts.min()
    nk, nx = t_amp.shape[1:]
    t2 = numpy.ascontiguousarray(t_amp).reshape(nstate * nk, nx)
    for tq, wq in zip(quad.t, quad.w):
        m = numpy.zeros((nx, nx), dtype=dtype)
        for delta, v in blocks:
            f = numpy.exp(-tq * (delta + c_act - c_shift))
            m += la.gemm(v, v * f[:, None], 'C', 'N')           # M_q[X, Y]
        g = numpy.exp(-tq * (lam - c_act))                      # (nk,)
        tm = la.gemm(t2, m, 'N', 'T')                           # (M_q T^J[k])[X]
        left = (t_amp * g[None, :, None]).reshape(nstate, nk * nx)
        # contr[I, J] = sum_kX g_k conj(T^I[k, X]) (M_q T^J[k])[X]
        tm = tm.reshape(nstate, nk * nx)
        f = wq * numpy.exp(tq * (shifts - c_shift))
        if col is None:
            a -= la.gemm(tm, left, 'N', 'C').T * f[None, :]
        else:
            a[:, col] -= la.gemm(tm[col:col + 1], left, 'N', 'C')[0] * f[col]
    return a


def laplace_class(pclass, ctx, act, civecs, nelec, shifts, nquad=None, kind='de'):
    '''a[I, J] = <I|V R(E0_J) V|J> for one class via the Laplace transform.'''
    nstate = numpy.asarray(civecs).shape[1]
    blocks = engine.class_blocks(pclass, ctx)
    if not blocks:
        return numpy.zeros((nstate, nstate), dtype=numpy.asarray(civecs).dtype)
    t_amp, sector = engine.transition_amplitudes(pclass, act, civecs, nelec)
    if t_amp is None:
        return numpy.zeros((nstate, nstate), dtype=numpy.asarray(civecs).dtype)
    return laplace_contract(blocks, t_amp, sector.lam, shifts, nquad, kind, pclass.name)

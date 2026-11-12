#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
'''Block Davidson for the Hermitian TDA problem A x = w x with locking.

pyscf.tdscf._lr_eig.eigh adds MAX_SPACE_INC (20) trial vectors every iteration
whatever the number of unconverged roots, forms residuals for up to max_space -
len(subspace) Ritz vectors (hundreds when max_memory is large) and QR-factorises
and orthogonalises all of them, and preconditions every root with the lowest
eigenvalue.  For the two-component spinor TDA (BiPh3, DYVTZ-SF, nocc nvir =
2.9e5 complex amplitudes, 10 roots) that was 1500 s of 4200 s outside the
matvecs, and 20 matvec vectors per iteration to the end.

Here only the nroots Ritz vectors are formed, a root whose residual norm is below
tol is locked (it keeps its Ritz vector in the subspace but gets no new trial
vector), every unconverged root is preconditioned with its own eigenvalue, the new
vectors are orthogonalised twice against the subspace, and the subspace collapses
to the 2 nroots lowest Ritz vectors when it exceeds max_space.  The batch of an
iteration is the number of unconverged roots.

    conv, e, x = eigh(aop, x0, precond, nroots, tol=1e-5, lindep=1e-12, max_cycle=100, max_space=None,
                      pick=None, verbose=None)

aop(xs) -> A xs for rows xs; precond(r, e) the preconditioned residual; pick(w, v)
-> (w, v) the admissible Ritz pairs (e.g. positive), ascending.
'''

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger


def _orth_against(xt, xs, lindep):
    '''xt orthonormalised against the rows of xs (twice, classical Gram-Schmidt) and among themselves; rows below
    lindep dropped'''
    out = []
    for x in xt:
        # relative: the preconditioned residuals of nearly converged roots are tiny
        n0 = numpy.linalg.norm(x)
        if n0 == 0:
            continue
        x = x / n0
        for _ in range(2):
            if len(xs):
                x = x - (xs.conj() @ x) @ xs
            for y in out:
                x = x - (y.conj() @ x) * y
        nrm = numpy.linalg.norm(x)
        if nrm ** 2 > lindep:
            out.append(x / nrm)
    return numpy.asarray(out)


def eigh(aop, x0, precond, nroots, tol=1e-5, lindep=1e-12, max_cycle=100, max_space=None, pick=None,
         verbose=None, log=None):
    if log is None:
        log = logger.new_logger(None, verbose if verbose is not None else logger.WARN)
    x0 = numpy.asarray(x0)
    if x0.ndim == 1:
        x0 = x0[None]
    dtype = numpy.result_type(x0.dtype, numpy.complex128)
    if max_space is None:
        max_space = max(8 * nroots, 40)
    xs = _orth_against(numpy.asarray(x0, dtype=dtype), numpy.zeros((0, x0.shape[1]), dtype), lindep)
    ax = numpy.asarray(aop(xs), dtype=dtype)
    conv = numpy.zeros(nroots, dtype=bool)
    e = None
    xr = None
    nmatvec = len(xs)
    for icyc in range(max_cycle):
        heff = xs.conj() @ ax.T
        heff = (heff + heff.conj().T) * .5
        w, v = scipy.linalg.eigh(heff)
        if pick is not None:
            w, v = pick(w, v)
        if len(w) < nroots:
            raise RuntimeError('not enough admissible Ritz roots (%d < %d)' % (len(w), nroots))
        elast = e
        e = w[:nroots]
        vr = v[:, :nroots]
        xr = vr.T @ xs
        axr = vr.T @ ax
        r = axr - e[:, None] * xr
        rn = numpy.linalg.norm(r, axis=1)
        conv = rn < tol
        de = e - elast if elast is not None and elast.size == e.size else e
        log.debug('lr_locked %d: subspace %d, |r| max %.3g, %d/%d converged, max|de| %.3g', icyc, len(xs),
                  rn.max(), conv.sum(), nroots, abs(de).max())
        if conv.all():
            break
        new = numpy.asarray([precond(r[k], e[k]) for k in range(nroots) if not conv[k]])
        if len(xs) + len(new) > max_space:
            # collapse to the lowest Ritz vectors (2 nroots of the admissible ones, at least the nroots)
            # (orthonormal: the eigenvectors of the Hermitian subspace matrix are)
            keep = min(len(w), 2 * nroots)
            vk = v[:, :keep]
            xs = vk.T @ xs
            ax = vk.T @ ax
            log.debug('lr_locked: subspace collapsed to %d', len(xs))
        new = _orth_against(new, xs, lindep)
        if len(new) == 0:
            log.debug('lr_locked: no new trial vector (linear dependence)')
            break
        axn = numpy.asarray(aop(new), dtype=dtype)
        nmatvec += len(new)
        xs = numpy.vstack([xs, new])
        ax = numpy.vstack([ax, axn])
    log.debug('lr_locked: %d matvec vectors', nmatvec)
    return conv, e, xr

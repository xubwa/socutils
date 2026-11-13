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


def _orth_against(xt, xs, lindep, real=False):
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
                c = xs.conj() @ x
                x = x - (c.real if real else c) @ xs
            for y in out:
                c = y.conj() @ x
                x = x - (c.real if real else c) * y
        nrm = numpy.linalg.norm(x)
        if nrm ** 2 > lindep:
            out.append(x / nrm)
    return numpy.asarray(out)


STALL_CYCLES = 4


def eigh(aop, x0, precond, nroots, tol=1e-5, lindep=1e-12, max_cycle=100, max_space=None, pick=None,
         verbose=None, log=None, project=None, real=False, stall_de=1e-11):
    '''project(xs) maps new trial vectors into an invariant subspace (e.g. the time-reversal even one); real=True:
    the subspace is a real vector space (real coefficients, the subspace matrix taken real), as for the TR-even
    vectors of a Kramers-restricted reference, whose overlaps and A matrix elements are real.'''
    if log is None:
        log = logger.new_logger(None, verbose if verbose is not None else logger.WARN)
    x0 = numpy.asarray(x0)
    if x0.ndim == 1:
        x0 = x0[None]
    dtype = numpy.result_type(x0.dtype, numpy.complex128)
    if max_space is None:
        max_space = max(8 * nroots, 40)
    x0 = numpy.asarray(x0, dtype=dtype)
    if project is not None:
        x0 = project(x0)
    xs = _orth_against(x0, numpy.zeros((0, x0.shape[1]), dtype), lindep, real)
    ax = numpy.asarray(aop(xs), dtype=dtype)
    conv = numpy.zeros(nroots, dtype=bool)
    stalled = None
    e = None
    xr = None
    nmatvec = len(xs)
    for icyc in range(max_cycle):
        heff = xs.conj() @ ax.T
        heff = (heff + heff.conj().T) * .5
        if real:
            heff = heff.real
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
        # a residual floor: a matvec that is not exactly Hermitian (COSX K: the one-sided overlap fit) cannot
        # bring |r| below its asymmetry; a root whose eigenvalue no longer moves (|de| < stall_de over
        # STALL_CYCLES iterations) is accepted there (logged)
        stall = numpy.abs(de) < stall_de if elast is not None and elast.size == e.size else numpy.zeros(nroots, bool)
        stalled = stalled + 1 if stalled is not None else numpy.zeros(nroots, int)
        stalled[~stall] = 0
        floor = (stalled >= STALL_CYCLES) & ~conv
        if floor.any():
            log.info('lr_locked: %d roots at a residual floor (|r| %s), accepted', floor.sum(),
                     ' '.join('%.1e' % x for x in rn[floor]))
            conv = conv | floor
        log.debug('lr_locked %d: subspace %d, |r| max %.3g, %d/%d converged, max|de| %.3g', icyc, len(xs),
                  rn.max(), conv.sum(), nroots, abs(de).max())
        if conv.all():
            break
        new = numpy.asarray([precond(r[k], e[k]) for k in range(nroots) if not conv[k]])
        if project is not None:
            new = project(new)
        if len(xs) + len(new) > max_space:
            # collapse to the lowest Ritz vectors (2 nroots of the admissible ones, at least the nroots)
            # (orthonormal: the eigenvectors of the Hermitian subspace matrix are)
            keep = min(len(w), 2 * nroots)
            vk = v[:, :keep]
            xs = vk.T @ xs
            ax = vk.T @ ax
            log.debug('lr_locked: subspace collapsed to %d', len(xs))
        nwant = len(new)
        new = _orth_against(new, xs, lindep, real)
        if len(new) < nwant:
            # degenerate roots: their preconditioned residuals (same diagonal shift) can be linearly dependent;
            # complete with the plain residuals
            rr = numpy.asarray([r[k] for k in range(nroots) if not conv[k]])
            if project is not None:
                rr = project(rr)
            extra = _orth_against(rr, numpy.vstack([xs, new]) if len(new) else xs, lindep, real)
            if len(extra):
                new = numpy.vstack([new, extra[:nwant - len(new)]]) if len(new) else extra[:nwant]
        if len(new) == 0:
            log.debug('lr_locked: no new trial vector (linear dependence)')
            break
        axn = numpy.asarray(aop(new), dtype=dtype)
        nmatvec += len(new)
        xs = numpy.vstack([xs, new])
        ax = numpy.vstack([ax, axn])
    log.debug('lr_locked: %d matvec vectors', nmatvec)
    return conv, e, xr

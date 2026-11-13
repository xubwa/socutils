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
    if not out:
        return numpy.zeros((0, xs.shape[1]), xs.dtype)
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


# ---------------------------------------------------------------------------------------------------------------
# RPA: (A + B)(X + Y) = w (X - Y), (A - B)(X - Y) = w (X + Y) with real symmetric A +- B

class RPAInstability(RuntimeError):
    '''A + B or A - B is not positive definite on the subspace (interlacing: nor on the full space), i.e. the
    reference is unstable (w^2 < 0 roots); the RPA has no real positive root there'''


def _chol(m, name):
    try:
        return scipy.linalg.cholesky(m, lower=True)
    except numpy.linalg.LinAlgError:
        mu = numpy.linalg.eigvalsh(m)
        raise RPAInstability('%s is not positive definite: lowest subspace eigenvalue %.6g (%d negative of %d)'
                             % (name, mu[0], (mu <= 0).sum(), len(mu)))


def _sub(xs, ys, real):
    m = xs.conj() @ ys.T
    return m.real if real else m


def rpa_eig(aop, x0, precond, nroots, tol=1e-5, lindep=1e-12, max_cycle=100, max_space=None, pick=None,
            verbose=None, log=None, project=None, stall_de=1e-11):
    '''Davidson for the real (time-reversal adapted) RPA with separate subspaces for P = X + Y and Q = X - Y
    (Bai & Li, SIAM J. Matrix Anal. Appl. 33, 1075 (2012); the subspace problem is the Stratmann-Scuseria-Frisch
    one, symmetric definite):

        (A + B) P = w Q,   (A - B) Q = w P,   P^T Q = 1   (|X|^2 - |Y|^2 = 1)

    aop(ps, qs) -> ((A + B) ps, (A - B) qs) for equally many rows ps, qs: one response density per pair (the
    transition density of X = (p + q) / 2, Y = (p - q) / 2), so a pair costs one TDA matvec.  On the subspaces
    P = Bp p, Q = Bq q (orthonormal rows) with M+ = Bp^T (A + B) Bp = L L^T, M- = Bq^T (A - B) Bq = R R^T and
    S = Bp^T Bq the stationary points of the RPA functional (p^T M+ p + q^T M- q) / 2 p^T S q are the singular
    triplets of C = L^-1 S R^-T: w = 1 / sigma, p = L^-T u / sqrt(sigma), q = R^-T v / sqrt(sigma).  Each
    unconverged root adds one P and one Q vector (the residuals preconditioned with (D - w)^-1 for X, (D + w)^-1
    for Y); locking, the residual floor, the collapse to 2 nroots and the degenerate top-up as in eigh.
    precond(r, e) -> r / (D - e).  The vectors may be complex (the TR-even amplitudes of a Kramers-restricted
    reference), the subspace is a real vector space: real coefficients, the real part of the overlaps.  Raises
    RPAInstability when M+ or M- is not positive definite.

    Returns conv, w, P, Q (rows; X = (P + Q) / 2, Y = (P - Q) / 2).'''
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
    real = True
    bp = _orth_against(x0, numpy.zeros((0, x0.shape[1]), dtype), lindep, real)
    bq = bp.copy()
    hp, hq = aop(bp, bq)
    hp = numpy.asarray(hp, dtype=dtype)
    hq = numpy.asarray(hq, dtype=dtype)
    nmatvec = len(bp)
    stalled = None
    e = P = Q = None
    conv = numpy.zeros(nroots, dtype=bool)
    for icyc in range(max_cycle):
        mp = _sub(bp, hp, real)
        mm = _sub(bq, hq, real)
        mp = (mp + mp.conj().T) * .5
        mm = (mm + mm.conj().T) * .5
        s = _sub(bp, bq, real)
        lp = _chol(mp, 'A + B')
        lm = _chol(mm, 'A - B')
        c = scipy.linalg.solve_triangular(lp, s, lower=True)
        c = scipy.linalg.solve_triangular(lm, c.conj().T, lower=True).conj().T          # L^-1 S R^-T
        u, sig, vh = numpy.linalg.svd(c, full_matrices=False)
        keep = sig > 1e-14 * sig.max()
        u, sig, v = u[:, keep], sig[keep], vh[keep].conj().T
        w = 1. / sig                                     # ascending
        cp = scipy.linalg.solve_triangular(lp.conj().T, u, lower=False) / numpy.sqrt(sig)
        cq = scipy.linalg.solve_triangular(lm.conj().T, v, lower=False) / numpy.sqrt(sig)
        if pick is not None:
            idx = pick(w)
            w, cp, cq = w[idx], cp[:, idx], cq[:, idx]
        if len(w) < nroots:
            raise RuntimeError('not enough admissible Ritz roots (%d < %d)' % (len(w), nroots))
        elast = e
        e = w[:nroots]
        P = cp[:, :nroots].T @ bp
        Q = cq[:, :nroots].T @ bq
        rp = cp[:, :nroots].T @ hp - e[:, None] * Q          # (A + B) P - w Q
        rq = cq[:, :nroots].T @ hq - e[:, None] * P          # (A - B) Q - w P
        rn = numpy.sqrt((numpy.linalg.norm(rp, axis=1) ** 2 + numpy.linalg.norm(rq, axis=1) ** 2) * .5)
        conv = rn < tol
        same = elast is not None and elast.size == e.size
        de = e - elast if same else e
        stall = numpy.abs(de) < stall_de if same else numpy.zeros(nroots, bool)
        stalled = stalled + 1 if stalled is not None else numpy.zeros(nroots, int)
        stalled[~stall] = 0
        floor = (stalled >= STALL_CYCLES) & ~conv
        if floor.any():
            log.info('lr_locked.rpa: %d roots at a residual floor (|r| %s), accepted', floor.sum(),
                     ' '.join('%.1e' % x for x in rn[floor]))
            conv = conv | floor
        log.debug('lr_locked.rpa %d: subspace %d + %d, |r| max %.3g, %d/%d converged, max|de| %.3g', icyc,
                  len(bp), len(bq), rn.max(), conv.sum(), nroots, abs(de).max())
        if conv.all():
            break
        todo = numpy.where(~conv)[0]
        # X residual (rp + rq) / 2 with (D - w)^-1, Y residual (rp - rq) / 2 with (D + w)^-1
        dx = [precond((rp[k] + rq[k]) * .5, e[k]) for k in todo]
        dy = [precond((rp[k] - rq[k]) * .5, -e[k]) for k in todo]
        newp = numpy.asarray([a + b for a, b in zip(dx, dy)])
        newq = numpy.asarray([a - b for a, b in zip(dx, dy)])
        if project is not None:
            newp = project(newp)
            newq = project(newq)
        if len(bp) + len(newp) > max_space or len(bq) + len(newq) > max_space:
            # collapse onto the lowest Ritz vectors (2 nroots), orthonormalised: Bp <- (QR of the coefficients)^T Bp
            nk = min(len(w), 2 * nroots)
            qp = numpy.linalg.qr(cp[:, :nk])[0]
            qq = numpy.linalg.qr(cq[:, :nk])[0]
            bp, hp = qp.T @ bp, qp.T @ hp
            bq, hq = qq.T @ bq, qq.T @ hq
            log.debug('lr_locked.rpa: subspaces collapsed to %d', nk)
        nwant = len(newp)
        newp = _orth_against(newp, bp, lindep, real)
        newq = _orth_against(newq, bq, lindep, real)
        # degenerate roots: complete with the plain residuals ((A - B) Q - w P lies in the P space)
        if len(newp) < nwant:
            extra = _orth_against(project(rq[todo]) if project is not None else rq[todo],
                                  numpy.vstack([bp, newp]) if len(newp) else bp, lindep, real)
            if len(extra):
                newp = numpy.vstack([newp, extra[:nwant - len(newp)]]) if len(newp) else extra[:nwant]
        if len(newq) < nwant:
            extra = _orth_against(project(rp[todo]) if project is not None else rp[todo],
                                  numpy.vstack([bq, newq]) if len(newq) else bq, lindep, real)
            if len(extra):
                newq = numpy.vstack([newq, extra[:nwant - len(newq)]]) if len(newq) else extra[:nwant]
        if len(newp) == 0 and len(newq) == 0:
            log.debug('lr_locked.rpa: no new trial vector (linear dependence)')
            break
        npair = max(len(newp), len(newq))
        n = bp.shape[1]
        zp = numpy.zeros((npair, n), dtype)
        zq = numpy.zeros((npair, n), dtype)
        zp[:len(newp)] = newp
        zq[:len(newq)] = newq
        hpn, hqn = aop(zp, zq)
        nmatvec += npair
        bp = numpy.vstack([bp, newp])
        hp = numpy.vstack([hp, numpy.asarray(hpn, dtype=dtype)[:len(newp)]])
        bq = numpy.vstack([bq, newq])
        hq = numpy.vstack([hq, numpy.asarray(hqn, dtype=dtype)[:len(newq)]])
    log.debug('lr_locked.rpa: %d response densities', nmatvec)
    return conv, e, P, Q

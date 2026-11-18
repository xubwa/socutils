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


def _extra_tda(k, x, t, e, precond):
    '''further directions of root k (Ritz vector x, correction t = (D - e)^-1 r): (D - e)^-1 x (with t it spans
    the Olsen correction), then (D - e)^-1 t'''
    yield precond(x, e)
    yield precond(t, e)


def _fill(new, todo, rn, extras, min_batch):
    '''new extended to min_batch rows by the extra directions of the unconverged roots todo, round robin from
    the largest residual'''
    if min_batch is None or len(new) >= min_batch or len(todo) == 0:
        return new
    gens = [extras(k) for k in sorted(todo, key=lambda k: -rn[k])]
    out = list(new)
    while len(out) < min_batch and gens:
        for g in list(gens):
            if len(out) >= min_batch:
                break
            try:
                out.append(next(g))
            except StopIteration:
                gens.remove(g)
    return numpy.asarray(out)


REFINE_KEEP = 1.        # the restart on the final operator from REFINE_KEEP x nroots Ritz vectors
REFINE_STALL = 3        # the cheaper operator is left also when max |r| has not halved for this many iterations
REFINE_STALL_FROM = 10.  # (counted once max |r| is below REFINE_STALL_FROM x rtol)


def _refine_due(st, rn, conv, rtol):
    '''whether to switch to the final operator: every residual below rtol, or the residual stalled (the cheaper
    operator's own floor, e.g. the COSX K of a coarse grid is less symmetric)'''
    if (conv | (rn < rtol)).all():
        return True
    r = rn[~conv].max()
    if r > REFINE_STALL_FROM * rtol:
        # still far (e.g. roots still being found, |r| flat for several iterations): no stall
        st['best'], st['since'] = numpy.inf, 0
        return False
    if r < .5 * st.get('best', numpy.inf):
        st['best'], st['since'] = r, 0
    else:
        st['since'] = st.get('since', 0) + 1
    return st['since'] >= REFINE_STALL


def eigh(aop, x0, precond, nroots, tol=1e-5, lindep=1e-12, max_cycle=100, max_space=None, pick=None,
         verbose=None, log=None, project=None, real=False, stall_de=1e-11, min_batch=None, refine=None):
    '''project(xs) maps new trial vectors into an invariant subspace (e.g. the time-reversal even one); real=True:
    the subspace is a real vector space (real coefficients, the subspace matrix taken real), as for the TR-even
    vectors of a Kramers-restricted reference, whose overlaps and A matrix elements are real.

    min_batch: when fewer roots are unconverged, each adds further directions ((D - e)^-1 x, (D - e)^-2 r) until
    the batch has min_batch vectors (the fixed cost of a response call spread over more vectors).
    refine = (rtol, switch[, tol2]): a cheaper operator first (e.g. the COSX response K on a coarse grid); once every
    residual is below rtol (or max |r| stalls, REFINE_STALL), switch() returns the final operator, the subspace
    restarts from the Ritz vectors and the solve converges to tol (tol2 if given) on it.  tol may be an array
    (per root).'''
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
    rst = {}
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
        if refine is not None and _refine_due(rst, rn, conv, refine[0]):
            # the final operator: restart from the Ritz vectors
            aop = refine[1]()
            if len(refine) > 2:
                tol = refine[2]
            refine = None
            nk = min(len(w), int(round(REFINE_KEEP * nroots)))
            log.debug('lr_locked: operator switched, restart from %d Ritz vectors', nk)
            xs = _orth_against(v[:, :nk].T @ xs, numpy.zeros((0, xs.shape[1]), dtype), lindep, real)
            ax = numpy.asarray(aop(xs), dtype=dtype)
            nmatvec += len(xs)
            e = stalled = None
            continue
        if conv.all():
            break
        todo = [k for k in range(nroots) if not conv[k]]
        new = numpy.asarray([precond(r[k], e[k]) for k in todo])
        if min_batch is not None and len(new) < min_batch:
            tk = dict(zip(todo, new))
            new = _fill(new, todo, rn, lambda k: _extra_tda(k, xr[k], tk[k], e[k], precond), min_batch)
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


def _pair_aop(aop, ps, qs, dtype):
    '''((A + B) ps, (A - B) qs, number of pairs) for possibly unequally many rows (padded with zeros)'''
    npair = max(len(ps), len(qs))
    n = ps.shape[1] if len(ps) else qs.shape[1]
    zp = numpy.zeros((npair, n), dtype)
    zq = numpy.zeros((npair, n), dtype)
    zp[:len(ps)] = ps
    zq[:len(qs)] = qs
    hp, hq = aop(zp, zq)
    return (numpy.asarray(hp, dtype=dtype)[:len(ps)], numpy.asarray(hq, dtype=dtype)[:len(qs)], npair)


def rpa_eig(aop, x0, precond, nroots, tol=1e-5, lindep=1e-12, max_cycle=100, max_space=None, pick=None,
            verbose=None, log=None, project=None, stall_de=1e-11, min_batch=None, refine=None):
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
    RPAInstability when M+ or M- is not positive definite.  min_batch, refine as in eigh (the extra pairs from
    (D -+ w)^-1 applied to X, Y and to the corrections).

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
    rst = {}
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
        if refine is not None and _refine_due(rst, rn, conv, refine[0]):
            aop = refine[1]()
            if len(refine) > 2:
                tol = refine[2]
            refine = None
            nk = min(len(w), int(round(REFINE_KEEP * nroots)))
            log.debug('lr_locked.rpa: operator switched, restart from %d Ritz vectors', nk)
            z = numpy.zeros((0, P.shape[1]), dtype)
            bp = _orth_against(cp[:, :nk].T @ bp, z, lindep, real)
            bq = _orth_against(cq[:, :nk].T @ bq, z, lindep, real)
            hp, hq, npair = _pair_aop(aop, bp, bq, dtype)
            nmatvec += npair
            e = stalled = None
            continue
        if conv.all():
            break
        todo = numpy.where(~conv)[0]
        # X residual (rp + rq) / 2 with (D - w)^-1, Y residual (rp - rq) / 2 with (D + w)^-1
        dx = [precond((rp[k] + rq[k]) * .5, e[k]) for k in todo]
        dy = [precond((rp[k] - rq[k]) * .5, -e[k]) for k in todo]
        if min_batch is not None and len(todo) < min_batch:
            dd = {k: (a, b) for k, a, b in zip(todo, dx, dy)}

            def extras(k):
                x, y = (P[k] + Q[k]) * .5, (P[k] - Q[k]) * .5
                yield precond(x, e[k]), precond(y, -e[k])
                yield precond(dd[k][0], e[k]), precond(dd[k][1], -e[k])
            pairs = _fill(list(zip(dx, dy)), todo, rn, extras, min_batch)
            dx, dy = pairs[:, 0], pairs[:, 1]
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
        hpn, hqn, npair = _pair_aop(aop, newp, newq, dtype)
        nmatvec += npair
        bp = numpy.vstack([bp, newp])
        hp = numpy.vstack([hp, hpn])
        bq = numpy.vstack([bq, newq])
        hq = numpy.vstack([hq, hqn])
    log.debug('lr_locked.rpa: %d response densities', nmatvec)
    return conv, e, P, Q

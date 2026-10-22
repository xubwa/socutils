'''
Block Davidson for the lowest roots of a large Hermitian matrix.

The trial vectors and their images are rows of two arrays, so that Ritz
vectors, residuals and orthogonalizations are threaded matrix products;
on a restart the images of the Ritz vectors are combined from the stored
images instead of being recomputed.
'''

import os
import tempfile
import time

import numpy

from socutils.zmr.lib import linalg as la

last = {}


def _orthonormalize(new, basis, lindep):
    '''Rows of new, orthogonalized against the rows of basis and among
    themselves (twice); rows that vanish are dropped.'''
    out = []
    for x in new:
        for _ in range(2):
            if len(basis):
                # coefficients <b_k|x> = conj((B x^+)_k): no conjugated copy of B
                x = x - la.gemm(la.gemm(basis, x[None], 'N', 'C').conj().T, basis)[0]
            for y in out:
                x = x - numpy.vdot(y, x) * y
        norm = numpy.linalg.norm(x)
        if norm > lindep:
            out.append(x / norm)
    return out


def scratch_dir():
    '''Directory of the scratch files: ZMR_TMPDIR, else PySCF's lib.param.TMPDIR
    (PYSCF_TMPDIR / TMPDIR).'''
    d = os.environ.get('ZMR_TMPDIR')
    if d:
        return d
    from pyscf import lib
    return lib.param.TMPDIR


def davidson(aop, x0, precond, nroots, tol=5e-5, max_cycle=100, max_space=12,
             lindep=1e-12, callback=None, follow=False, incore=None):
    '''Lowest nroots eigenpairs.

    aop(X): H applied to the rows of X (n, dim) -> (n, dim)
    x0: initial vectors, at least nroots
    precond(r, e): approximate (H - e)^-1 r
    Converged when the residual norm |H x - e x| of every root (|x| = 1) is
    below tol; the error of the energy is of second order in it.
    max_space: trial vectors per root before a restart.
    follow: the roots are the nroots Ritz vectors with the largest weight in
    the span of the initial vectors (root following by maximum overlap)
    instead of the lowest ones: states that are not the lowest of the
    operator, e.g. the lowest of one symmetry when the symmetry is not
    imposed on the vectors.
    callback(icyc, space, e, de, res_norms, seconds)

    incore: bytes the trial vectors and their images (2 x (max_space x nroots
    + nroots) vectors) may take in memory; above it they are kept in mapped
    scratch files (scratch_dir()), which the operating system pages in and
    out, so that a long expansion does not exhaust the memory.  None: always
    in memory.

    Returns (converged, energies, vectors as rows).  davidson.last holds
    the seconds spent in aop, in the Ritz step, in the preconditioner and in
    the orthogonalization of the last call.'''
    global last
    last = {'aop': 0.0, 'ritz': 0.0, 'precond': 0.0, 'orth': 0.0}

    def timed(key, fn, *args):
        t = time.perf_counter()
        out = fn(*args)
        last[key] += time.perf_counter() - t
        return out

    dim = len(x0[0])
    limit = max(max_space * nroots, 2 * nroots + 2)
    shape = limit + nroots, dim
    tmp = None
    last['on_disk'] = incore is not None and 2 * 16.0 * shape[0] * shape[1] > incore
    if last['on_disk']:
        tmp = tempfile.TemporaryDirectory(prefix='zmr_davidson_', dir=scratch_dir())
        v = numpy.memmap(os.path.join(tmp.name, 'v'), dtype=numpy.complex128, mode='w+',
                         shape=shape)
        w = numpy.memmap(os.path.join(tmp.name, 'w'), dtype=numpy.complex128, mode='w+',
                         shape=shape)
    else:
        v = numpy.empty(shape, dtype=numpy.complex128)
        w = numpy.empty(shape, dtype=numpy.complex128)
    start = _orthonormalize([numpy.asarray(x, dtype=numpy.complex128) for x in x0],
                            numpy.empty((0, dim)), lindep)
    n = len(start)
    v[:n] = start
    w[:n] = timed('aop', aop, v[:n])
    if follow:
        # overlaps <reference_r|v_i> of the trial vectors with the initial ones
        ref = v[:n].copy()
        proj = numpy.zeros((n, limit + nroots), dtype=numpy.complex128)
        proj[:, :n] = la.gemm(ref, v[:n], 'N', 'C').conj()
    e_old = None
    conv = False
    for icyc in range(max_cycle):
        t0 = time.perf_counter()
        # Rayleigh-Ritz in the span of the n trial vectors
        # <v_i|w_j> = conj((V W^+)_ij): no conjugated copy of the vectors
        tr = time.perf_counter()
        h = la.gemm(v[:n], w[:n], 'N', 'C').conj()
        e, y = numpy.linalg.eigh(0.5 * (h + h.conj().T))
        if follow:
            weight = (abs(proj[:, :n] @ y)**2).sum(axis=0)
            pick = numpy.sort(numpy.argsort(-weight, kind='stable')[:nroots])
        else:
            pick = numpy.arange(nroots)
        e, y = e[pick], numpy.ascontiguousarray(y[:, pick].T)            # (nroots, n)
        x = la.gemm(y, v[:n])
        ax = la.gemm(y, w[:n])
        r = ax - e[:, None] * x
        last['ritz'] += time.perf_counter() - tr
        norms = numpy.linalg.norm(r, axis=1)
        de = abs(e - e_old) if e_old is not None and len(e_old) == len(e) else numpy.full(
            len(e), numpy.inf)
        e_old = e
        done = norms < tol
        if callback is not None:
            callback(icyc, n, e, de, norms, time.perf_counter() - t0)
        if done.all():
            conv = True
            break
        if n + nroots > limit:
            # restart from the Ritz vectors; their images are known
            v[:nroots], w[:nroots] = x, ax
            n = nroots
            if follow:
                proj[:, :n] = la.gemm(ref, v[:n], 'N', 'C').conj()
        new = [timed('precond', precond, r[k], e[k]) for k in range(nroots) if not done[k]]
        new = timed('orth', _orthonormalize, new, v[:n], lindep)
        if not new:
            break
        k = len(new)
        v[n:n + k] = new
        if follow:
            proj[:, n:n + k] = la.gemm(ref, v[n:n + k], 'N', 'C').conj()
        w[n:n + k] = timed('aop', aop, v[n:n + k])
        n += k
    e, x = numpy.array(e), numpy.array(x)
    del v, w
    if tmp is not None:
        tmp.cleanup()
    return conv, e, x

'''
The eight perturber classes of UC-NEVPT2 (docs/derivations/uc-nevpt2.md).

For every class, the part of V|Psi_J> = Q H |Phi_c Psi_J> in that class is

    sum_e E_e sum_X v[e, X] O_X |Phi_c Psi_J>

with E_e a string of core / virtual operators (external label e), O_X a
string of active operators and v[e, X] a combination of integrals.  The
functions E_e |Phi_c k> are orthonormal eigenfunctions of H_D with eigenvalue
E_core + lambda_k + delta_e, delta_e = sum(eps particles) - sum(eps holes).

The active strings are non-redundant: two creators (or two annihilators)
carry an ordered pair of indices, so no O_X is a multiple of another one.

A class supplies
    dn              change of the active electron number
    blocks(ctx)     iterator over (delta[e], v[e, X]) for blocks of external labels
    operators(nact) the strings O_X as blocks (daggers, orbital indices),
                    see ActiveSpace.apply_strings
    apply(act, c, nelec)  O_X |c> for CI vectors c (..., dim_N), returned as
                    (..., nX, dim_{N+dn})
    subset(nact)    position of every O_X among the strings with unrestricted
                    indices (the layout of zmr.wick.pc_nevpt2.OPERATORS)
Only these pieces are class specific; the contraction engine is shared.
'''

import copy

import numpy

from socutils.zmr.lib import linalg as la


def _antisym(eris, labels):
    '''<pr||qs> = (pq|rs) - (ps|rq) as an array indexed [p, r, q, s] with the
    spaces of p, r, q, s given by labels.'''
    p, r, q, s = labels
    return (eris.eri_block(p + q + r + s).transpose(0, 2, 1, 3)
            - eris.eri_block(p + s + r + q).transpose(0, 2, 3, 1))


def _pairs(n):
    return numpy.triu_indices(n, k=1)


class ExternalBlocks:
    '''External blocks of a class that are not listed up front.

    nlabel: number of external labels; delta_min: smallest external energy;
    w_nodes(d): W[m][X, Y] = sum_e conj(v[e, X]) v[e, Y] / (d[m] + delta_e)
    for an array of energy arguments, straight from the three-index
    integrals (native kernels, one task per core orbital or pair), or None.
    Iterating gives the blocks (delta, v) themselves, built on first use and
    kept: the form the other contraction routes work with.'''

    dtype = numpy.complex128

    def __init__(self, make, nlabel, delta_min, w_nodes):
        self._make = make
        self._list = None
        self.nlabel = nlabel
        self.delta_min = delta_min
        self.w_nodes = w_nodes

    def materialize(self):
        if self._list is None:
            self._list = [(numpy.asarray(d), numpy.asarray(v)) for d, v in self._make()]
        return self._list

    def __bool__(self):
        return self.nlabel > 0

    def __iter__(self):
        return iter(self.materialize())

    def __getitem__(self, key):
        return self.materialize()[key]


def _external(pclass, ctx, nlabel, delta_min, kernel):
    '''ExternalBlocks of a doubly external class if the integrals are
    factorized, else None.'''
    if _factors(ctx.eris) is None or nlabel <= 0:
        return None

    def w_nodes(d):
        return kernel(numpy.asarray(d, dtype=float))

    return ExternalBlocks(lambda: pclass.blocks(ctx), nlabel, delta_min, w_nodes)


def _two_smallest(e):
    return numpy.sort(e)[:2].sum()


def _two_largest(e):
    return numpy.sort(e)[-2:].sum()


def _factors(eris):
    '''cd_block of a factorized integral source, else None.'''
    return eris.cd_block if getattr(eris, 'factorized', False) else None


def _as3(w):
    return None if w is None else w.reshape(len(w), 1, 1)


def _slice(b, k):
    '''B[:, :, k] of three-index integrals as a contiguous (naux, n) matrix.'''
    return numpy.ascontiguousarray(b[:, :, k])


class Context:
    '''Integrals shared by the class builders.'''

    def __init__(self, eris, eps_core, eps_virt, fcore):
        self.eris = eris
        self.sp = eris.spaces
        self.eps_c = numpy.asarray(eps_core)
        self.eps_v = numpy.asarray(eps_virt)
        self.fcore = fcore           # core Fock h + sum_i <pi||qi>
        self.keep = None             # filter of core labels, see core_filter


def core_filter(ctx, keep):
    '''Context in which the ccaa, ccav and caaa blocks contain only the core
    labels for which keep(name, ii, jj) is true: ii, jj arrays of the core
    orbitals of the pairs i < j (ccaa, ccav) or ii the single core orbitals
    and jj None (caaa).  The other classes are not affected.'''
    new = copy.copy(ctx)
    new.keep = keep
    return new


def _kept(ctx, name, ii, jj=None):
    if getattr(ctx, 'keep', None) is None:
        return numpy.ones(len(ii), dtype=bool)
    return numpy.asarray(ctx.keep(name, ii, jj), dtype=bool)


class _CoreSubset:
    '''Integral blocks with the core index restricted to the orbitals idx.'''

    def __init__(self, eris, idx):
        from socutils.zmr.spaces import OrbitalSpaces
        sp = eris.spaces
        self._eris = eris
        self._idx = idx
        self.spaces = OrbitalSpaces(len(idx), sp.nact, sp.nvirt)

    def eri_block(self, labels):
        out = self._eris.eri_block(labels)
        for axis, label in enumerate(labels):
            if label == 'c':
                out = out.take(self._idx, axis=axis)
        return out

    @property
    def factorized(self):
        return getattr(self._eris, 'factorized', False)

    def cd_block(self, pq):
        out = self._eris.cd_block(pq)
        for axis, label in enumerate(pq):
            if label == 'c':
                out = out.take(self._idx, axis=axis + 1)
        return out


def core_subset(ctx, idx):
    '''Context in which only the core orbitals idx take part: the others
    are neither excited nor removed from the mean field.'''
    sp = ctx.sp
    idx = numpy.asarray(idx, dtype=int)
    keep = numpy.concatenate([idx, numpy.arange(sp.ncore, sp.nmo)])
    return Context(_CoreSubset(ctx.eris, idx), ctx.eps_c[idx], ctx.eps_v,
                   ctx.fcore[numpy.ix_(keep, keep)])


def _all(nact, n):
    '''All n-tuples of active indices as rows, last index fastest (the
    single empty tuple for n = 0).'''
    if n == 0:
        return numpy.zeros((1, 0), dtype=numpy.int64)
    return numpy.indices((nact,) * n).reshape(n, -1).T


class PerturberClass:
    name = None
    dn = 0

    def blocks(self, ctx):
        raise NotImplementedError

    def operators(self, nact):
        raise NotImplementedError

    def apply(self, act, c, nelec):
        return act.apply_strings(c, nelec, self.operators(act.norb))

    def subset(self, nact):
        return None


class CCVV(PerturberClass):
    '''E_e = a^+ b^+ j i (i<j, a<b), O = 1, v = <ab||ij>'''
    name, dn = 'ccvv', 0

    def blocks(self, ctx):
        nc, nv = ctx.sp.ncore, ctx.sp.nvirt
        if nc < 2 or nv < 2:
            return
        cd = _factors(ctx.eris)
        ii, jj = _pairs(nc)
        aa, bb = _pairs(nv)
        if cd is None:
            g = _antisym(ctx.eris, 'vvcc')       # [a, b, i, j]
        else:
            bvc = cd('vc')
        for i, j in zip(ii, jj):
            if cd is None:
                v = g[aa, bb, i, j][:, None]
            else:
                # (ai|bj) for this pair of core orbitals only; <ab||ij> = (ai|bj) - (aj|bi)
                m = la.gemm(_slice(bvc, i), _slice(bvc, j), 'T', 'N')
                v = (m[aa, bb] - m[bb, aa])[:, None]
            delta = ctx.eps_v[aa] + ctx.eps_v[bb] - ctx.eps_c[i] - ctx.eps_c[j]
            yield delta, v

    def external(self, ctx):
        nc, nv = ctx.sp.ncore, ctx.sp.nvirt
        if nc < 2 or nv < 2:
            return None
        from socutils.zmr.lib import kernels
        return _external(
            self, ctx, nc * (nc - 1) // 2 * (nv * (nv - 1) // 2),
            _two_smallest(ctx.eps_v) - _two_largest(ctx.eps_c),
            lambda d: _as3(kernels.w_ccvv(ctx.eris.cd_block('vc'), ctx.eps_v, ctx.eps_c, d)))

    def operators(self, nact):
        return [((), _all(nact, 0))]


class CCAV(PerturberClass):
    '''E_e = a^+ j i (i<j), O_t = t^+, v = <at||ij>'''
    name, dn = 'ccav', 1

    def blocks(self, ctx):
        nc, nv = ctx.sp.ncore, ctx.sp.nvirt
        if nc < 2 or nv < 1:
            return
        # the block (vc|ac) has one virtual index only: small, taken whole
        g = _antisym(ctx.eris, 'vacc')           # [a, t, i, j]
        ii, jj = _pairs(nc)
        keep = _kept(ctx, 'ccav', ii, jj)
        for i, j in zip(ii[keep], jj[keep]):
            v = g[:, :, i, j]                     # (a, t)
            delta = ctx.eps_v - ctx.eps_c[i] - ctx.eps_c[j]
            yield delta, v

    def operators(self, nact):
        return [((True,), _all(nact, 1))]


class CAVV(PerturberClass):
    '''E_e = a^+ b^+ i (a<b), O_t = t, v = -<ab||it>'''
    name, dn = 'cavv', -1

    def blocks(self, ctx):
        nc, nv = ctx.sp.ncore, ctx.sp.nvirt
        if nc < 1 or nv < 2:
            return
        cd = _factors(ctx.eris)
        aa, bb = _pairs(nv)
        if cd is None:
            g = _antisym(ctx.eris, 'vvca')       # [a, b, i, t]
        else:
            bvc = cd('vc')
            bva = numpy.ascontiguousarray(cd('va'))
            bva = bva.reshape(bva.shape[0], -1)   # (L, b t)
        for i in range(nc):
            if cd is None:
                v = -g[aa, bb, i, :]              # (ab, t)
            else:
                # <ab||it> = (ai|bt) - (at|bi) for this core orbital only
                m = la.gemm(_slice(bvc, i), bva, 'T', 'N').reshape(nv, nv, -1)
                v = -(m[aa, bb] - m[bb, aa])
            delta = ctx.eps_v[aa] + ctx.eps_v[bb] - ctx.eps_c[i]
            yield delta, v

    def external(self, ctx):
        nc, nv = ctx.sp.ncore, ctx.sp.nvirt
        if nc < 1 or nv < 2 or ctx.sp.nact < 1:
            return None
        from socutils.zmr.lib import kernels
        return _external(
            self, ctx, nc * (nv * (nv - 1) // 2), _two_smallest(ctx.eps_v) - ctx.eps_c.max(),
            lambda d: kernels.w_cavv(ctx.eris.cd_block('vc'), ctx.eris.cd_block('va'),
                                     ctx.eps_v, ctx.eps_c, d))

    def operators(self, nact):
        return [((False,), _all(nact, 1))]


class CCAA(PerturberClass):
    '''E_e = j i (i<j), O_tu = t^+ u^+ (t<u), v = <tu||ij>'''
    name, dn = 'ccaa', 2

    def blocks(self, ctx):
        nc, na = ctx.sp.ncore, ctx.sp.nact
        if nc < 2 or na < 2:
            return
        g = _antisym(ctx.eris, 'aacc')           # [t, u, i, j]
        ii, jj = _pairs(nc)
        tt, uu = _pairs(na)
        keep = _kept(ctx, 'ccaa', ii, jj)
        if not keep.any():
            return
        ii, jj = ii[keep], jj[keep]
        v = g[tt, uu][:, ii, jj].T               # (ij, tu)
        delta = -ctx.eps_c[ii] - ctx.eps_c[jj]
        yield delta, v

    def operators(self, nact):
        tt, uu = _pairs(nact)
        return [((True, True), numpy.stack([tt, uu], axis=1))]

    def subset(self, nact):
        tt, uu = _pairs(nact)
        return tt * nact + uu


class AAVV(PerturberClass):
    '''E_e = a^+ b^+ (a<b), O_tu = u t (t<u), v = <ab||tu>'''
    name, dn = 'aavv', -2

    def blocks(self, ctx):
        nv, na = ctx.sp.nvirt, ctx.sp.nact
        if nv < 2 or na < 2:
            return
        g = _antisym(ctx.eris, 'vvaa')           # [a, b, t, u]
        aa, bb = _pairs(nv)
        tt, uu = _pairs(na)
        v = g[aa, bb][:, tt, uu]                 # (ab, tu)
        delta = ctx.eps_v[aa] + ctx.eps_v[bb]
        yield delta, v

    def external(self, ctx):
        nv, na = ctx.sp.nvirt, ctx.sp.nact
        if nv < 2 or na < 2:
            return None
        from socutils.zmr.lib import kernels
        return _external(self, ctx, nv * (nv - 1) // 2, _two_smallest(ctx.eps_v),
                         lambda d: kernels.w_aavv(ctx.eris.cd_block('va'), ctx.eps_v, d))

    def operators(self, nact):
        tt, uu = _pairs(nact)
        return [((False, False), numpy.stack([uu, tt], axis=1))]

    def subset(self, nact):
        tt, uu = _pairs(nact)
        return tt * nact + uu


class CAVA(PerturberClass):
    '''E_e = a^+ i, O = {1, t^+ u}, v = {f_ai, <at||iu>}'''
    name, dn = 'cava', 0

    def blocks(self, ctx):
        sp = ctx.sp
        nc, nv, na = sp.ncore, sp.nvirt, sp.nact
        if nc < 1 or nv < 1:
            return
        g = _antisym(ctx.eris, 'vaca')           # [a, t, i, u]
        f_vc = ctx.fcore[sp.slice('v'), sp.slice('c')]
        for i in range(nc):
            v = numpy.concatenate([f_vc[:, i, None],
                                   g[:, :, i, :].reshape(nv, na * na)], axis=1)
            delta = ctx.eps_v - ctx.eps_c[i]
            yield delta, v

    def operators(self, nact):
        return [((), _all(nact, 0)), ((True, False), _all(nact, 2))]


class CAAA(PerturberClass):
    '''E_e = i, O = {t^+, t^+ u^+ v (t<u)}, v = {-f_ti, -<tu||iv>}'''
    name, dn = 'caaa', 1

    def blocks(self, ctx):
        sp = ctx.sp
        nc, na = sp.ncore, sp.nact
        if nc < 1:
            return
        g = _antisym(ctx.eris, 'aaca')           # [t, u, i, v]
        f_ac = ctx.fcore[sp.slice('a'), sp.slice('c')]
        tt, uu = _pairs(na)
        v = numpy.concatenate([-f_ac.T,
                               -g[tt, uu].transpose(1, 0, 2).reshape(nc, len(tt) * na)], axis=1)
        keep = _kept(ctx, 'caaa', numpy.arange(nc))
        if keep.any():
            yield -ctx.eps_c[keep], v[keep]

    def operators(self, nact):
        tt, uu = _pairs(nact)
        vv = numpy.arange(nact)
        ops = numpy.stack([numpy.repeat(tt, nact), numpy.repeat(uu, nact),
                           numpy.tile(vv, len(tt))], axis=1)            # X = (tu, v)
        return [((True,), _all(nact, 1)), ((True, True, False), ops)]

    def subset(self, nact):
        tt, uu = _pairs(nact)
        x3 = (tt * nact + uu)[:, None] * nact + numpy.arange(nact)
        return numpy.concatenate([numpy.arange(nact), nact + x3.ravel()])


class AAAV(PerturberClass):
    '''E_e = a^+, O = {t, u^+ v t (t<v)}, v = {f_at, <au||tv>}'''
    name, dn = 'aaav', -1

    def blocks(self, ctx):
        sp = ctx.sp
        nv, na = sp.nvirt, sp.nact
        if nv < 1:
            return
        g = _antisym(ctx.eris, 'vaaa')           # [a, u, t, v]
        f_va = ctx.fcore[sp.slice('v'), sp.slice('a')]
        tt, vv = _pairs(na)
        v = numpy.concatenate([f_va, g[:, :, tt, vv].reshape(nv, na * len(tt))], axis=1)
        yield ctx.eps_v.copy(), v

    def operators(self, nact):
        tt, vv = _pairs(nact)
        uu = numpy.arange(nact)
        ops = numpy.stack([numpy.repeat(uu, len(tt)), numpy.tile(vv, nact),
                           numpy.tile(tt, nact)], axis=1)                # X = (u, tv)
        return [((False,), _all(nact, 1)), ((True, False, False), ops)]

    def subset(self, nact):
        tt, vv = _pairs(nact)
        # unrestricted layout of u^+ v t: (t, u, v)
        x3 = tt[None, :] * nact**2 + numpy.arange(nact)[:, None] * nact + vv[None, :]
        return numpy.concatenate([numpy.arange(nact), nact + x3.ravel()])


ALL_CLASSES = (CCVV(), CCAV(), CAVV(), CCAA(), AAVV(), CAVA(), CAAA(), AAAV())
CLASS_BY_NAME = {c.name: c for c in ALL_CLASSES}

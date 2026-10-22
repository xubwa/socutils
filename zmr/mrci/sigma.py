'''
sigma = H C in the product basis from the terms of zmr.wick.mrci.

Every term is one einsum of an integral block, the ket coefficients and a
matrix of an active operator string between the block bases,

    Gamma[k', t.., u.., k] = <b'_k'| t^+ .. u .. |b_k>,

computed in the determinant basis of the active sectors and kept.  The
all-active part of H enters through <b'_k'| H_aa |b_k>, the constant
through the unit matrix.
'''

import string
import resource
import time

import numpy

from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.lib import kernels
from socutils.zmr.lib import linalg as la
from socutils.zmr.uc.classes import _all
from socutils.zmr.wick import mrci as terms


def _einsum(subs, ops, out):
    '''numpy.einsum for the setup; two operands without traces are one
    threaded matrix product between threaded transpositions.'''
    if len(ops) == 2 and all(len(set(x)) == len(x) for x in subs):
        asub, bsub = subs
        con = [x for x in asub if x in bsub and x not in out]
        aown = [x for x in asub if x not in con]
        bown = [x for x in bsub if x not in con]
        if (con and not set(aown) & set(bown) and set(aown + bown) == set(out)
                and len(out) == len(aown) + len(bown)):
            a, b = ops
            at = kernels.transposed(a, [asub.index(x) for x in aown + con])
            bt = kernels.transposed(b, [bsub.index(x) for x in con + bown])
            na = int(numpy.prod(at.shape[:len(aown)]))
            r = la.gemm(at.reshape(na, -1), bt.reshape(at.size // na, -1))
            r = r.reshape(at.shape[:len(aown)] + bt.shape[len(con):])
            order = aown + bown
            if order == list(out):
                return r
            return kernels.transposed(r, [order.index(x) for x in out])
    return numpy.einsum(','.join(subs) + '->' + out, *ops, optimize=True)


# seconds by step inside the contractions: {((bra, ket, kind), step): seconds}.  Sigma
# points _STEPS at its own table and sets the context before every contraction.
_STEPS = {}
_CONTEXT = [None]


def _step(name, t0):
    '''Charge the time since t0 to a step of the running contraction.'''
    now = time.perf_counter()
    key = _CONTEXT[0], name
    _STEPS[key] = _STEPS.get(key, 0.0) + now - t0
    return now


def _contig(a):
    '''C-contiguous complex array with the content of a; views (transposed
    integral blocks) are copied by the threaded kernel.'''
    if a.dtype == numpy.complex128 and a.flags.c_contiguous:
        return a
    return kernels.transposed(a, range(a.ndim))


def _antisymmetrized(w, x, y, scale=1.0):
    '''scale * (w - w with the axes x and y exchanged), as a new array.'''
    perm = list(range(w.ndim))
    perm[x], perm[y] = y, x
    w = numpy.asarray(w)
    if w.dtype != numpy.complex128:
        w = w.astype(numpy.complex128)
    out = numpy.empty(w.shape, dtype=numpy.complex128)
    kernels.transpose_add(out, w, range(w.ndim), alpha=scale, assign=True)
    return kernels.transpose_add(out, w, perm, alpha=-scale)


class _Shape:
    '''Stands for an array that is not formed.'''

    def __init__(self, shape):
        self.shape = shape


def _pair_flat(n):
    '''Flat positions of (i, j) and (j, i), i < j, in an n x n block.'''
    i, j = numpy.triu_indices(n, k=1)
    return i * n + j, j * n + i


def _pack_axes(a, first, n):
    '''a with the axes (first, first + 1), both of dimension n, replaced by
    one axis over the pairs i < j.'''
    return kernels.pair_gather(_contig(a), first)


def _unpack_axis(a, axis, n):
    '''The antisymmetric array whose pairs i < j along `axis` are a.'''
    return kernels.pair_scatter(_contig(a), axis, n)


def _unpack(r, axes):
    '''Unpack the pair axes (positions in r, ascending) one after the other.'''
    for ax in reversed(axes):
        n = int(round((1 + numpy.sqrt(1 + 8 * r.shape[ax])) / 2))
        r = _unpack_axis(r, ax, n)
    return r


def _collect(acc, total, r, perm, axes):
    '''Add the result r (pair axes `axes` still packed) to the results of
    the same layout, or to total at once if there is no accumulator.'''
    if acc is None:
        kernels.transpose_add(total, _unpack(r, axes), perm)
        return
    key = tuple(perm), tuple(axes), r.shape
    if key in acc:
        kernels.transpose_add(acc[key], r, range(r.ndim))
    else:
        acc[key] = r


class _Contraction:
    '''out[Z, bra.., L] += sum W[.., L, K] C[Z, ket.., K] as one matrix product:
    W is stored as the matrix (its own indices) x (contracted indices), so
    that only C and the result are transposed in an iteration.

    Independent labels only: C is antisymmetric in equivalent ket indices and
    the result is antisymmetrized in equivalent bra indices, so

    * a pair (x, y) of ket_pairs that is contracted is summed over x < y with
      W[x, y] - W[y, x];
    * a pair (A, B) of bra_pairs that W carries alone is stored for A < B as
      (W[A, B] - W[B, A]) / 2, and the product is scattered with both signs:
      that is the antisymmetrized contribution itself.'''

    def __init__(self, expr, w, bra_pairs=(), ket_pairs=()):
        ops, out = expr.split('->')
        wsub, csub = ops.split(',')
        self.expr, self.w = expr, None
        if set(wsub) & set(csub) & set(out):
            self.w = w                       # an index in all three: plain einsum
            return
        con = [x for x in csub if x in wsub]                 # contracted
        wown = [x for x in wsub if x not in con]
        cown = [x for x in csub if x not in con]
        # packing costs a gather of C and a scatter of the result in every
        # iteration: only for matrices large enough to be worth it
        if w.size < Sigma.pack_limit:
            bra_pairs = ket_pairs = ()
        kpack = [p for p in ket_pairs if p[0] in con and p[1] in con]
        bpack = [p for p in bra_pairs if p[0] in wown and p[1] in wown]
        letters = list(wsub)
        for x, y in kpack:
            w = _antisymmetrized(w, letters.index(x), letters.index(y))
        for x, y in bpack:
            w = _antisymmetrized(w, letters.index(x), letters.index(y), 0.5)

        def grouped(names, pairs):
            single = [x for x in names if not any(x in p for p in pairs)]
            return [(x,) for x in single] + [tuple(p) for p in pairs]

        own_groups = grouped(wown, bpack)
        con_groups = grouped(con, kpack)
        order = [x for g in own_groups + con_groups for x in g]
        w = w.transpose([wsub.index(x) for x in order])
        axis = 0
        for g in own_groups + con_groups:
            if len(g) == 2:
                w = _pack_axes(w, axis, w.shape[axis])
            axis += 1
        self.w_shape = w.shape[:len(own_groups)]
        w = _contig(w)
        self.mat = kernels.transposed(w.reshape(int(numpy.prod(self.w_shape)), -1), (1, 0))
        con_order = [x for g in con_groups for x in g]
        self.c_perm = [csub.index(x) for x in cown + con_order]
        self.ncon = len(con_order)
        # positions (among the contracted axes) of the packed ket pairs, and
        # (among the own axes of W) of the packed bra pairs
        self.kpos = [k for k, g in enumerate(con_groups) if len(g) == 2]
        self.bpos = [k for k, g in enumerate(own_groups) if len(g) == 2]
        own_order = [x for g in own_groups for x in g]
        self.ncown = len(cown)
        self.out_perm = [(cown + own_order).index(x) for x in out]

    def _c_matrix(self, c, cache):
        key = tuple(self.c_perm), tuple(self.kpos)
        if key not in cache:
            base = tuple(self.c_perm)
            if base not in cache:
                # one contiguous copy per permutation, whatever the split into
                # own and contracted indices
                cache[base] = kernels.transposed(c, self.c_perm)
            ct = cache[base]
            for k in self.kpos:                       # groups before k take one axis each
                ax = self.ncown + k
                ct = _pack_axes(ct, ax, ct.shape[ax])
            cache[key] = ct
        return cache[key]

    def add(self, c, total, work=None):
        '''work = (cache of transposed coefficient matrices of this ket,
        results by layout): contractions of one sigma share them, so that
        every layout is transposed once.'''
        t = time.perf_counter()
        if self.w is not None:
            total += numpy.einsum(self.expr, self.w, c, optimize=True)
            _step('einsum', t)
            return
        cache, acc = work if work is not None else ({}, None)
        ct = self._c_matrix(c, cache)
        t = _step('C transposed, packed', t)
        own = ct.shape[:self.ncown]
        r = la.gemm(ct.reshape(int(numpy.prod(own)), -1), self.mat)
        t = _step('product W C', t)
        r = r.reshape(own + self.w_shape)
        _collect(acc, total, r, self.out_perm, [self.ncown + k for k in self.bpos])
        _step('result collected', t)


def _flush(acc, total):
    for (perm, axes, _), r in acc.items():
        kernels.transpose_add(total, _unpack(r, axes), perm)
    acc.clear()


def _slices(c, total, spect, size, limit):
    '''Slices of C and of the result along a spectator index (one that
    passes through the term untouched), so that the intermediate of a
    two-step contraction stays below limit elements.'''
    axis_c, axis_o = spect
    n = c.shape[axis_c]
    step = max(1, int(n * limit / size))
    for k0 in range(0, n, step):
        sl = slice(k0, k0 + step)
        yield (numpy.ascontiguousarray(c[(slice(None),) * axis_c + (sl,)]),
               total[(slice(None),) * axis_o + (sl,)])


class _TwoStage:
    '''A term whose dressed integral would be large: the matrix Gamma acts on
    the coefficients first, X[Z, ket.., t.., L] = sum_K Gamma[L, t.., K] C[Z,
    ket.., K], then the integrals are contracted with X.

    With the indices of C ordered as (spectators S, contracted D), X comes
    out of the first product as [Z, S, (D, t..), L] and the second product
    is a stack of matrix products G[own, (D, t..)] X[(D, t..), L] over
    (Z, S): nothing is transposed but C itself and the result, and the
    stack is processed in slices that keep X below Sigma.x_limit.'''

    def __init__(self, gsub, gamma, g2, g, csub, out, bra_pairs=()):
        acts = gsub[1:-1]
        # Gamma[L, t.., K] as the matrix [K, (t.., L)]
        nd = gamma.ndim
        self.gmat = kernels.transposed(gamma, [nd - 1] + list(range(1, nd - 1)) + [0]).reshape(
            gamma.shape[-1], -1)
        self.nl = gamma.shape[0]
        self.nact = int(numpy.prod(gamma.shape[1:-1]))
        spect = [x for x in csub if x in out and x not in g2]
        dummy = [x for x in csub if x not in spect]
        own = [x for x in g2 if x not in dummy and x not in acts]
        if (set(own) - set(out)) or (set(dummy) - set(g2)) or (set(spect) & set(g2)):
            raise NotImplementedError('two-step contraction %s, %s' % (g2, csub))
        self.c_perm = [0] + [1 + csub.index(x) for x in spect + dummy] + [len(csub) + 1]
        self.nspect = len(spect)
        # an antisymmetric bra pair carried by the integrals alone: A < B only
        bpack = [p for p in bra_pairs if p[0] in own and p[1] in own]
        if g.size < Sigma.pack_limit:
            bpack = []
        for x, y in bpack:
            g = _antisymmetrized(g, g2.index(x), g2.index(y), 0.5)
        own = [x for x in own if not any(x in p for p in bpack)] + [x for p in bpack for x in p]
        gm = g.transpose([g2.index(x) for x in own + dummy + list(acts)])
        nsingle = len(own) - 2 * len(bpack)
        for k in range(len(bpack)):
            gm = _pack_axes(gm, nsingle + k, gm.shape[nsingle + k])
        self.bpos = [nsingle + k for k in range(len(bpack))]
        self.own_shape = gm.shape[:nsingle + len(bpack)]
        self.gmat2 = _contig(gm).reshape(int(numpy.prod(self.own_shape)), -1)
        self.out_perm = [(['Z'] + spect + own + ['L']).index(x) for x in 'Z' + out + 'L']
        # for the loop over the active indices: C as [D, Z, S, K], the result
        # as [own, Z, S, L]
        self.c_perm2 = self.c_perm[1 + len(spect):-1] + self.c_perm[:1 + len(spect)] + [
            self.c_perm[-1]]
        self.ndummy = len(dummy)
        self.out_perm2 = [(own + ['Z'] + spect + ['L']).index(x) for x in 'Z' + out + 'L']
        self.g3 = self.gam3 = None

    def _by_active_index(self, c, total, work):
        '''sum over the active indices a of G_a C Gamma_a: two matrix
        products of good shape per a, no intermediate larger than sigma.'''
        if self.g3 is None:
            nown, m = self.gmat2.shape[0], self.gmat.shape[0]
            self.g3 = kernels.transposed(self.gmat2.reshape(nown, -1, self.nact), (2, 0, 1))
            self.gam3 = kernels.transposed(self.gmat.reshape(m, self.nact, self.nl), (1, 0, 2))
            self.gmat = self.gmat2 = None
        cache = work[0] if work is not None else {}
        key = tuple(self.c_perm2)
        if key not in cache:
            cache[key] = kernels.transposed(c, self.c_perm2)
        ct = cache[key]
        dshape, rest = ct.shape[:self.ndummy], ct.shape[self.ndummy:-1]
        m = ct.shape[-1]
        nd = int(numpy.prod(dshape))
        cd = ct.reshape(nd, -1)
        nown = self.g3.shape[1]
        res = numpy.zeros((nown * (cd.shape[1] // m), self.nl), dtype=numpy.complex128)
        t = numpy.empty((nown, cd.shape[1]), dtype=numpy.complex128)
        for a in range(self.nact):
            # both products into existing arrays, the second one accumulating
            la.gemm(self.g3[a], cd, out=t)
            la.gemm(t.reshape(-1, m), self.gam3[a], out=res, beta=1.0)
        res = res.reshape(self.own_shape + rest + (self.nl,))
        _collect(work[1] if work is not None else None, total, res, self.out_perm2,
                 list(self.bpos))

    def add(self, c, total, work=None):
        if self.g3 is not None or (float(c.size) / c.shape[-1] * self.nact * self.nl
                                   > Sigma.loop_limit):
            return self._by_active_index(c, total, work)
        t = time.perf_counter()
        cache = work[0] if work is not None else {}
        key = tuple(self.c_perm)
        if key not in cache:
            cache[key] = kernels.transposed(c, self.c_perm)
        ct = cache[key]
        t = _step('C transposed', t)
        lead = ct.shape[:1 + self.nspect]
        m = ct.shape[-1]
        nrow = int(numpy.prod(lead))
        nd = ct.size // (nrow * m)
        ct = ct.reshape(nrow, nd, m)
        nown = self.gmat2.shape[0]
        res = numpy.empty((nrow, nown, self.nl), dtype=numpy.complex128)
        step = max(1, int(Sigma.x_limit // (nd * self.nact * self.nl)))
        for r0 in range(0, nrow, step):
            r1 = min(nrow, r0 + step)
            x = la.gemm(ct[r0:r1].reshape(-1, m), self.gmat)
            t = _step('product Gamma C', t)
            x = x.reshape(r1 - r0, nd * self.nact, self.nl)
            for r in range(r1 - r0):
                la.gemm(self.gmat2, x[r], out=res[r0 + r])
            t = _step('product G X', t)
        res = res.reshape(lead + self.own_shape + (self.nl,))
        _collect(work[1] if work is not None else None, total, res, self.out_perm,
                 [len(lead) + k for k in self.bpos])
        _step('result collected', t)


class _IntegralFirst:
    '''A term whose dressed integral would be large and whose integrals
    contract external indices of the coefficients: Y = G C first (the
    intermediate carries the remaining ket indices, the bra indices and the
    active indices), then Gamma over the active indices and K.'''

    def __init__(self, expr1, g, expr2, gamma, spect=None, bra_pairs=(), ket_pairs=()):
        self.first = _Contraction(expr1, g, bra_pairs, ket_pairs)
        self.second = _Contraction(expr2, gamma)
        ops, out = expr1.split('->')
        self.g_letters, self.c_letters, self.y_letters = ops.split(',') + [out]
        self.dims = dict(zip(self.g_letters, g.shape))
        self.spect = spect

    def _y(self, c):
        dims = dict(self.dims)
        dims.update(zip(self.c_letters, c.shape))
        return numpy.zeros([dims[x] for x in self.y_letters], dtype=numpy.complex128)

    def add(self, c, total, work=None):
        dims = dict(self.dims)
        dims.update(zip(self.c_letters, c.shape))
        size = numpy.prod([float(dims[x]) for x in self.y_letters])
        if size > Sigma.x_limit and self.spect is not None:
            for csl, tsl in _slices(c, total, self.spect, size, Sigma.x_limit):
                y = self._y(csl)
                self.first.add(csl, y)
                self.second.add(y, tsl)
            return
        y = self._y(c)
        self.first.add(c, y, (work[0], None) if work is not None else None)
        self.second.add(y, total, ({}, work[1]) if work is not None else None)


class _VVVV:
    '''out[Z, A, B, .., K] += coef sum_pq (Ap|Bq) C[Z, p, q, .., K] from the
    three-index integrals B[L, A, p] of the virtual orbitals, without the
    (vv|vv) block: for every remaining label x,

        R_x = sum_L B_L C_x B_L^T,

    two matrix products of cost naux nv^3.'''

    def __init__(self, coef, bvv):
        naux, nv, _ = bvv.shape
        self.coef = coef
        self.nv, self.naux = nv, naux
        # [A, L, p]: as ((A, L), p) for the first product, (B, (L, q)) for the second
        self.b = numpy.ascontiguousarray(bvv.transpose(1, 0, 2))

    def add(self, c, total, work=None):
        nv, naux = self.nv, self.naux
        # C[Z, p, q, rest.., K] -> [Z, rest.., K, p, q]
        ct = numpy.moveaxis(c, (1, 2), (-2, -1))
        lead = ct.shape[:-2]
        ct = numpy.ascontiguousarray(ct).reshape(-1, nv, nv)
        b1 = self.b.reshape(nv * naux, nv)
        b2 = self.b.reshape(nv, naux * nv)
        t = time.perf_counter()
        r = numpy.empty((len(ct), nv, nv), dtype=numpy.complex128)
        for x in range(len(ct)):
            if not ct[x].any():
                r[x] = 0
                continue
            m1 = la.gemm(b1, ct[x]).reshape(nv, naux * nv)
            r[x] = la.gemm(m1, b2, 'N', 'T')
        t = _step('products B C B^T, %d labels' % len(ct), t)
        total += self.coef * numpy.moveaxis(r.reshape(lead + (nv, nv)), (-2, -1), (1, 2))
        _step('result added', t)


class _VVVVDense:
    '''The same term with the matrix in memory, over independent labels:
    G[(A < B), (p < q)] = (Ap|Bq) - (Aq|Bp), built row block by row block
    from the three-index integrals (nothing else of that size is formed).
    With C antisymmetric in (p, q), sum_{p<q} G C[p, q] scattered with both
    signs is the antisymmetrized result: nv^4 / 4 per label x.'''

    def __init__(self, coef, bvv):
        naux, nv, _ = bvv.shape
        self.coef, self.nv = coef, nv
        pp, qq = numpy.triu_indices(nv, k=1)
        npair = len(pp)
        self.g = numpy.empty((npair, npair), dtype=numpy.complex128)
        bt = kernels.transposed(bvv, (1, 2, 0)).reshape(nv * nv, naux)     # [(B, q), L]
        row = 0
        for a in range(nv - 1):
            nb = nv - a - 1
            # (ap|Bq) for B > a only, as [B, q, p]
            m = la.gemm(bt[(a + 1) * nv:], _contig(bvv[:, a, :])).reshape(nb, nv, nv)
            t = kernels.transposed(m, (0, 2, 1))                           # [B, p, q]
            kernels.transpose_add(t, m, (0, 1, 2), alpha=-1.0)             # - (aq|Bp)
            self.g[row:row + nb] = kernels.pair_gather(t, 1)
            row += nb

    def add(self, c, total, work=None):
        nv, nd = self.nv, c.ndim
        # C[Z, p, q, rest.., K] -> [Z, rest.., K, (p < q)]
        t = time.perf_counter()
        ct = kernels.transposed(c, [0] + list(range(3, nd)) + [1, 2])
        lead = ct.shape[:-2]
        cp = kernels.pair_gather(ct, nd - 2)
        t = _step('C transposed, packed', t)
        r = la.gemm(cp.reshape(-1, cp.shape[-1]), self.g, 'N', 'T')
        t = _step('product (vv|vv) C', t)
        out = kernels.pair_scatter(r.reshape(lead + (r.shape[-1],)), nd - 2, nv)
        kernels.transpose_add(total, out, [0, nd - 2, nd - 1] + list(range(1, nd - 2)),
                              alpha=self.coef)
        _step('result unpacked, added', t)


class Sigma:
    # the (vv|vv) terms use the three-index integrals when the block would
    # need more than this many bytes (factorized integrals only); True / False
    # force one way
    direct_vvvv = None
    vvvv_memory = 2e9
    # ... and keep (vv|vv) as one matrix over pairs, built from them, up to this
    # many bytes (16 npair^2, about 4 nv^4); beyond it every label costs two
    # products of naux nv^3, about 40 times more in all for 300 virtuals
    vvvv_dense_memory = 4e10
    # dressed integrals with more elements than this are not formed
    w_limit = 2e7
    w_max = 5e8
    # intermediates of two-step contractions are sliced above this many
    # elements; matrices Gamma above gamma_limit elements are not stored
    x_limit = 2e8
    gamma_limit = 5e7
    # two-step contractions whose intermediate would exceed this many
    # elements loop over the active indices instead: no intermediate beyond
    # the size of sigma, but measured about 20% slower than the sliced form
    # (both are bound by memory traffic), so it is off by default
    loop_limit = float('inf')
    # vectors O_X b_k held at a time when matrices Gamma are built
    string_limit = 1e8
    # matrices with fewer elements are not packed over antisymmetric pairs
    pack_limit = 5e6

    def __init__(self, eris, act, basis, active_eri=None):
        self.eris = eris
        self.sp = eris.spaces
        self.act = act
        self.basis = basis
        sa = self.sp.slice('a')
        # all-active part of H with the bare one-electron integrals
        self.act_bare = ActiveSpace(eris.h1[sa, sa], eris.eri_block('aaaa')
                                    if active_eri is None else active_eri)
        self._terms = {}
        self._gamma = {}
        self._blocks = {}
        self._compiled = {}
        self._vvvv = None
        self.timings = {}
        self.wall, self.calls = 0.0, 0
        self.peak_growth, self._rss = {}, 0.0
        # seconds by step inside the contractions, see _step
        self.steps = {}
        # progress(text): called after every sigma vector that took more than
        # report_seconds, with the time, the peak memory and the slowest parts
        self.progress, self.report_seconds, self.report_lines = None, 30.0, 5
        self._dims = {'c': self.sp.ncore, 'a': self.sp.nact, 'v': self.sp.nvirt}

    def terms(self, bra, ket):
        if (bra, ket) not in self._terms:
            self._terms[bra, ket] = terms.reduced(bra, ket)
        return self._terms[bra, ket]

    def integral(self, name, idx):
        labels = ''.join(i.space for i in idx)
        key = name, labels
        if key not in self._blocks:
            if name == 'h':
                self._blocks[key] = self.eris.h1[tuple(self.sp.slice(x) for x in labels)]
            else:
                self._blocks[key] = numpy.asarray(self.eris.eri_block(labels))
        return self._blocks[key]

    def gamma(self, bra, ket, ncre, nann):
        '''<b'_k'| t_1^+ .. t_n^+ u_1 .. u_m |b_k> as [k', t.., u.., k];
        ncre = 'H': the all-active part of H, [k', k].'''
        key = bra, ket, ncre, nann
        if key not in self._gamma:
            bb, kb = self.basis.blocks[bra], self.basis.blocks[ket]
            na = self.sp.nact
            if ncre == 'H':
                g = bb.b.conj().T @ self.act_bare.apply_h(kb.nelec, kb.b)
            elif ncre + nann == 0:
                g = bb.b.conj().T @ kb.b
            else:
                g = self._gamma_dense(bb, kb, ncre, nann)
            self._gamma[key] = g
        return self._gamma[key]

    def _gamma_dense(self, bb, kb, ncre, nann):
        '''Gamma[k', X, k] = <b'_k'|O_X|b_k> for all strings X.  The strings
        act on the block with fewer functions (on the bra functions as the
        adjoint strings, <b'|O|b> = conj(<b|O^+|b'>)), in batches of
        functions that keep the vectors O_X b below string_limit elements.'''
        na, rank = self.sp.nact, ncre + nann
        daggers = (True,) * ncre + (False,) * nann
        ops = _all(na, rank)
        on_bra = bb.m < kb.m
        if on_bra:
            # O^+ = c_n^+ ... c_1^+
            daggers = tuple(not d for d in daggers[::-1])
            ops = numpy.ascontiguousarray(numpy.asarray(ops).reshape(len(ops), rank)[:, ::-1])
            src, dst = bb, kb
        else:
            src, dst = kb, bb
        proj = numpy.ascontiguousarray(dst.b.conj())
        g = numpy.empty((src.m, len(ops), dst.m), dtype=numpy.complex128)
        step = max(1, int(self.string_limit // (len(ops) * dst.b.shape[0])))
        for k0 in range(0, src.m, step):
            vec = numpy.ascontiguousarray(src.b[:, k0:k0 + step].T)
            out = self.act.apply_strings(vec, src.nelec, [(daggers, ops)])
            nk = out.shape[0]
            g[k0:k0 + nk] = la.gemm(out.reshape(nk * len(ops), -1), proj).reshape(nk, len(ops), dst.m)
            del out
        if on_bra:
            numpy.conjugate(g, out=g)                        # [k', X, k]
        else:
            g = kernels.transposed(g, (2, 1, 0))
        return g.reshape((bb.m,) + (na,) * rank + (kb.m,))

    def _operands(self, term, bra, ket, gamma):
        '''(subscripts of C, of the output, list of (subscript, array) of
        the other factors) of one term; letters: Z vector, K / L basis
        functions of ket / bra.'''
        free = terms.free_indices(bra)
        letters = iter('mnopqrstuvwxyz')
        names = {}
        for x in free:                                       # bra indices keep their names
            names[x] = x
        csub = None

        def sub(indices):
            for i in indices:
                if i.name not in names:
                    names[i.name] = next(letters)
            return ''.join(names[i.name] for i in indices)

        for name, idx in term.tensors:                       # C first: canonical dummy letters
            if name == 'C':
                csub = sub(idx)
        rest = []
        for name, idx in term.tensors:
            if name != 'C':
                rest.append((sub(idx), self.integral(name, idx)))
        for a, b in term.deltas:
            rest.append((sub((a, b)), numpy.eye(self._dims[a.space])))
        cre, ann = term.rdm
        rest.append(('L' + sub(cre) + sub(ann) + 'K', gamma))
        return csub, ''.join(free), rest

    def _compile(self, bra, ket):
        '''Contractions of one block pair.  The factors of a term that do
        not depend on the coefficients are contracted once, terms of the same
        index structure are added, and sigma += W C is one matrix product
        per structure.  Terms without active operators are a pure integral
        tensor times a matrix M[L, K] (the overlap of the block bases, or
        H_aa): the integral tensor is kept alone and M is applied to the sum
        of these contributions.  Returns ([contractions with L, K],
        [(M or None, [contractions without])]).'''
        derived, over = self.terms(bra, ket)
        full, ext = {}, {'S': {}, 'H': {}}
        jobs = [(t, 'S') for t in derived] + [(t, 'H') for t in over]
        direct = []
        two = {}
        streamed = {}
        for t, kind in jobs:
            cre, ann = t.rdm
            plain = not cre and not ann
            if kind == 'S' and self._use_direct_vvvv() and self._is_vvvv(t, bra, ket):
                # bytes of the matrix over pairs A < B, p < q that is kept
                npair = self.sp.nvirt * (self.sp.nvirt - 1) // 2
                dense = (self.direct_vvvv is None
                         and 16.0 * npair**2 <= self.vvvv_dense_memory)
                if dense and self._vvvv is None:
                    self._vvvv = _VVVVDense(1.0, self.eris.cd_block('vv'))
                if dense:
                    shared = _VVVVDense.__new__(_VVVVDense)
                    shared.coef, shared.nv, shared.g = complex(t.coef), self.sp.nvirt, self._vvvv.g
                    direct.append(shared)
                else:
                    direct.append(_VVVV(complex(t.coef), self.eris.cd_block('vv')))
                continue
            rank = len(cre) + len(ann)
            bb, kb = self.basis.blocks[bra], self.basis.blocks[ket]
            big = (kind == 'S' and rank
                   and float(bb.m) * kb.m * self.sp.nact**rank > self.gamma_limit)
            if big:
                g = _Shape((bb.m,) + (self.sp.nact,) * rank + (kb.m,))
            else:
                g = (self.gamma(bra, ket, len(cre), len(ann)) if kind == 'S'
                     else self.gamma(bra, ket, 'H', 0))
            csub, out, rest = self._operands(t, bra, ket, g)
            if plain:
                rest = rest[:-1]
            inside = set(''.join(x for x, _ in rest))
            # indices W shares with C or with the result
            wsub = ''.join(x for x in out + ''.join(sorted(set(csub) - set(out))) if x in inside)
            dims = {}
            for x, a in rest:
                dims.update(zip(x, a.shape))
            wsize = 0 if plain else numpy.prod([float(dims[x]) for x in wsub + 'LK'])
            if big and wsize <= self.w_max:
                # Gamma is not stored: W is accumulated from slices of it
                streamed.setdefault((len(cre), len(ann)), []).append(
                    (complex(t.coef), rest, wsub, '%sLK,Z%sK->Z%sL' % (wsub, csub, out), dims))
                continue
            if big:
                g = self.gamma(bra, ket, len(cre), len(ann))
                rest[-1] = (rest[-1][0], g)
            if wsize > self.w_limit:
                gsub, gam = rest[-1]
                acts = gsub[1:-1]
                csize = numpy.prod([float(dims.get(x, 0) or self._letter_dim(x, t, bra, ket))
                                    for x in csub])
                asize = numpy.prod([float(dims[x]) for x in acts])
                osize = numpy.prod([float(dims.get(x, 0) or self._letter_dim(x, t, bra, ket))
                                    for x in out])
                gamma_first = csize * asize * gam.shape[0]
                int_first = osize * asize * gam.shape[-1]
                # a dressed integral no larger than the intermediate it saves
                # in every iteration is formed after all
                if wsize <= self.w_max and wsize <= 4 * min(gamma_first, int_first):
                    wsize = 0
            if wsize > self.w_limit:
                ints = rest[:-1]
                inner = set(''.join(x for x, _ in ints))
                g2 = ''.join(x for x in out + ''.join(sorted(set(csub + acts) - set(out)))
                             if x in inner)
                g = complex(t.coef) * numpy.einsum(','.join(x for x, _ in ints) + '->' + g2,
                                                   *[a for _, a in ints], optimize=True)
                # the smaller intermediate: Gamma first or integrals first
                # an index of C that goes to the result untouched
                spect = [(1 + csub.index(x), 1 + out.index(x))
                         for x in csub if x in out and x not in g2]
                spect = spect[0] if spect else None
                if int_first < gamma_first:
                    key = ('int', '%s,Z%sK->Z%s%sK' % (g2, csub, out, acts), id(gam),
                           '%s,Z%s%sK->Z%sL' % (gsub, out, acts, out), spect)
                else:
                    key = ('gam', gsub, id(gam), g2, spect, csub, out)
                if key in two:
                    two[key][1] += g
                else:
                    two[key] = [gam, g]
                continue
            if rest:
                # (a single operand comes back as a view of the integrals: no
                # in-place scaling)
                w = _einsum([x for x, _ in rest], [a for _, a in rest],
                            wsub + ('' if plain else 'LK'))
                if (w.dtype == numpy.complex128
                        and not any(numpy.may_share_memory(w, a) for _, a in rest)):
                    w *= complex(t.coef)
                else:
                    w = complex(t.coef) * w
            else:
                w = numpy.asarray(complex(t.coef))
            if plain:
                acc, expr = ext[kind], '%s,Z%sK->Z%sK' % (wsub, csub, out)
            else:
                acc, expr = full, '%sLK,Z%sK->Z%sL' % (wsub, csub, out)
            if expr in acc:
                acc[expr] += w
            else:
                acc[expr] = w

        def build(d):
            return [_Contraction(expr, w, *self._pairs(bra, ket, expr))
                    for expr, w in d.items() if w.any()]

        staged = []
        for kind in ('S', 'H'):
            cons = build(ext[kind])
            if not cons:
                continue
            m = self.gamma(bra, ket, 0, 0) if kind == 'S' else self.gamma(bra, ket, 'H', 0)
            if kind == 'S' and bra == ket:
                m = None                    # orthonormal basis
            staged.append((m, cons))
        if direct:
            staged.append((None, direct))
        for (ncre, nann), items in streamed.items():
            for expr, w in self._streamed(bra, ket, ncre, nann, items).items():
                full[expr] = full[expr] + w if expr in full else w
        stages = [(_TwoStage(k[1], gam, k[3], g, k[5], k[6], self._pairs(bra, ket, None)[0])
                   if k[0] == 'gam'
                   else _IntegralFirst(k[1], g, k[3], gam, k[4], *self._pairs(bra, ket, k[1])))
                  for k, (gam, g) in two.items() if g.any()]
        return build(full) + stages, staged

    @staticmethod
    def _pairs(bra, ket, expr):
        '''(antisymmetric pairs of bra index letters, of the letters of C in
        the contraction expr) of a block pair.'''
        def pairs(letters, block):
            nh, np_ = terms.BLOCKS[block]
            out = []
            if np_ == 2:
                out.append((letters[0], letters[1]))
            if nh == 2:
                out.append((letters[np_], letters[np_ + 1]))
            return out
        bra_pairs = pairs(terms.free_indices(bra), bra)
        if expr is None:
            return bra_pairs, []
        csub = expr.split('->')[0].split(',')[1][1:-1]
        nh, np_ = terms.BLOCKS[ket]
        ket_pairs = [p for p in pairs(csub[:nh + np_], ket) if p[0] != p[1]]
        return bra_pairs, ket_pairs

    def _string_slices(self, bra, ket, ncre, nann):
        '''(k0, k1, O_X b_k as [k0:k1, X, determinant]) over slices of the
        ket basis.'''
        bb, kb = self.basis.blocks[bra], self.basis.blocks[ket]
        daggers = (True,) * ncre + (False,) * nann
        ops = _all(self.sp.nact, ncre + nann)
        step = max(1, int(self.gamma_limit // (len(ops) * bb.b.shape[0])))
        for k0 in range(0, kb.m, step):
            vec = numpy.ascontiguousarray(kb.b[:, k0:k0 + step].T)
            out = self.act.apply_strings(vec, kb.nelec, [(daggers, ops)])
            yield k0, k0 + out.shape[0], out

    def _streamed(self, bra, ket, ncre, nann, items):
        '''Dressed integrals W[w, L, K] = sum_X G[w, X] Gamma[L, X, K] of the
        terms items = [(coef, factors, wsub, expr, dims)] that share a Gamma
        too large to store.  With n_w values of the external indices w of W
        and m functions in the bra block, the integrals are contracted with
        the vectors O_X b_K before these are projected on the bra functions
        when n_w < m (cost n_w / m of the other order).'''
        bb = self.basis.blocks[bra]
        na, rank = self.sp.nact, ncre + nann
        bc = numpy.ascontiguousarray(bb.b.conj())
        out, first, last = {}, {}, []
        for coef, rest, wsub, expr, dims in items:
            if expr not in out:
                out[expr] = numpy.zeros([dims[x] for x in wsub + 'LK'], dtype=numpy.complex128)
            nw = int(numpy.prod([dims[x] for x in wsub]))
            if nw < bb.m and len(rest) > 1:
                g = coef * numpy.einsum(','.join(x for x, _ in rest[:-1]) + '->' + wsub
                                        + rest[-1][0][1:-1], *[a for _, a in rest[:-1]],
                                        optimize=True)
                g = g.reshape(nw, na**rank)
                first[expr] = first[expr] + g if expr in first else g
            else:
                last.append((coef, rest, wsub, expr))
        for k0, k1, vec in self._string_slices(bra, ket, ncre, nann):
            nk, nx, dim = vec.shape
            if first:
                xv = kernels.transposed(vec, (1, 0, 2)).reshape(nx, nk * dim)
            for expr, g in first.items():
                w = out[expr]
                # [w, k, determinant] -> [w, k, L]
                t = la.gemm(g, xv).reshape(len(g) * nk, dim)
                t = la.gemm(t, bc).reshape(len(g), nk, bb.m)
                w.reshape(len(g), bb.m, -1)[:, :, k0:k1] += t.transpose(0, 2, 1)
            xv = None
            if last:
                g = la.gemm(vec.reshape(nk * nx, dim), bc).reshape(nk, nx, bb.m)
                g = g.transpose(2, 1, 0).reshape((bb.m,) + (na,) * rank + (nk,))
                for coef, rest, wsub, expr in last:
                    subs = [x for x, _ in rest]
                    ops = [a for _, a in rest[:-1]] + [g]
                    out[expr][..., k0:k1] += coef * _einsum(subs, ops, wsub + 'LK')
        return out

    def _letter_dim(self, letter, term, bra, ket):
        '''Dimension of an index letter that occurs in C or the result only.'''
        free = terms.free_indices(bra)
        if letter in free:
            return self._dims['v' if letter in 'AB' else 'c']
        for name, idx in term.tensors:
            if name == 'C':
                nh, np_ = terms.BLOCKS[ket]
                # canonical dummy letters were assigned in the order of C
                pos = [k for k, i in enumerate(idx) if i.name not in free]
                letters = 'mnopqrstuvwxyz'
                k = pos[letters.index(letter)] if letter in letters[:len(pos)] else None
                if k is not None:
                    return self._dims['v' if k < np_ else 'c']
        raise KeyError(letter)

    def _use_direct_vvvv(self):
        if not getattr(self.eris, 'factorized', False):
            return False
        if self.direct_vvvv is not None:
            return self.direct_vvvv
        return 16.0 * self.sp.nvirt**4 > self.vvvv_memory

    @staticmethod
    def _is_vvvv(term, bra, ket):
        '''The term sum_pq (Ap|Bq) C[p, q, ...] of a doubly external block.'''
        if bra != ket or term.deltas or term.rdm != ((), ()) or len(term.tensors) != 2:
            return False
        tens = dict(term.tensors)
        if 'eri' not in tens or any(i.space != 'v' for i in tens['eri']):
            return False
        a, p, b, q = tens['eri']
        free = terms.free_indices(bra)
        c = tens['C']
        ok = ((a.name, b.name) == tuple(free[:2]) and (c[0], c[1]) == (p, q)
              and [i.name for i in c[2:]] == free[2:])
        if not ok:
            raise NotImplementedError('unexpected form of the (vv|vv) term: %r' % term)
        return True

    def compiled(self, bra, ket):
        key = bra, ket
        if key not in self._compiled:
            self._compiled[key] = self._compile(bra, ket)
        return self._compiled[key]

    def __call__(self, c):
        '''{block: sigma[z, a, b, i, j, k]} for coefficients {block: C[z, ...]}
        with a leading axis z over vectors.'''
        global _STEPS
        _STEPS = self.steps
        out = {}
        # transposed coefficients: one copy per ket and permutation for the
        # whole sigma; results: accumulated per bra and permutation
        caches = {ket: {} for ket in self.basis.names}
        for bra in self.basis.names:
            total = numpy.zeros(c[bra].shape, dtype=numpy.complex128)
            acc = {}
            for ket in self.basis.names:
                full, staged = self.compiled(bra, ket)
                cache = caches[ket]
                for con in full:
                    t0 = time.perf_counter()
                    _CONTEXT[0] = bra, ket, type(con).__name__
                    con.add(c[ket], total, (cache, acc))
                    self._tick(bra, ket, con, t0)
                for m, cons in staged:
                    if m is None:
                        for con in cons:
                            t0 = time.perf_counter()
                            _CONTEXT[0] = bra, ket, type(con).__name__
                            con.add(c[ket], total, (cache, acc))
                            self._tick(bra, ket, con, t0)
                        continue
                    # the matrix M[L, K] acts on the last index only.  It is applied
                    # where the intermediate is smaller: to the coefficients first,
                    # C'[.., L] = sum_K M[L, K] C[.., K] (ket labels x bra functions),
                    # or to the contracted result last (bra labels x ket functions).
                    t0 = time.perf_counter()
                    _CONTEXT[0] = bra, ket, 'with matrix M'
                    ck = c[ket]
                    nl, nk = m.shape
                    first = float(ck.size // ck.shape[-1]) * nl
                    last = float(total.size // total.shape[-1]) * nk
                    if last <= first:
                        part = numpy.zeros(total.shape[:-1] + (nk,), dtype=numpy.complex128)
                        pacc = {}
                        for con in cons:
                            con.add(ck, part, (cache, pacc))
                        _flush(pacc, part)
                        ts = time.perf_counter()
                        r = la.gemm(part.reshape(-1, nk), m, 'N', 'T').reshape(total.shape)
                        kernels.transpose_add(total, r, range(total.ndim))
                        _step('product result M', ts)
                        del part, r
                        self._tick(bra, ket, 'with matrix M', t0)
                        continue
                    # (C' is used by this pair of blocks only and released right
                    # after; in slices of the bra functions L that keep it below
                    # x_limit elements)
                    cmat = _contig(ck).reshape(-1, ck.shape[-1])
                    step = max(1, int(self.x_limit // len(cmat)))
                    for l0 in range(0, nl, step):
                        ml = m if step >= nl else numpy.ascontiguousarray(m[l0:l0 + step])
                        ts = time.perf_counter()
                        cm = la.gemm(cmat, ml, 'N', 'T').reshape(ck.shape[:-1] + (len(ml),))
                        _step('product C M', ts)
                        if step >= nl:
                            mcache = {}
                            for con in cons:
                                con.add(cm, total, (mcache, acc))
                        else:
                            part = numpy.zeros(total.shape[:-1] + (len(ml),),
                                               dtype=numpy.complex128)
                            mcache, pacc = {}, {}
                            for con in cons:
                                con.add(cm, part, (mcache, pacc))
                            _flush(pacc, part)
                            total[..., l0:l0 + step] += part
                        del cm, mcache
                    self._tick(bra, ket, 'with matrix M', t0)
            t0 = time.perf_counter()
            _CONTEXT[0] = bra, 'flush', ''
            _flush(acc, total)
            self._tick(bra, 'flush', None, t0)
            # the terms are representatives: antisymmetrize in the bra indices
            t0 = time.perf_counter()
            out[bra] = self.basis.blocks[bra].antisymmetrize(total) + self.eris.ecore * c[bra]
            self._tick(bra, 'antisymmetrize', None, t0)
        return out

    def compile_all(self):
        '''Prepare the contractions of all block pairs and release what was
        only needed for that: the cached matrices Gamma, the integral blocks
        and the four-index blocks of the integral source.'''
        for bra in self.basis.names:
            for ket in self.basis.names:
                self.compiled(bra, ket)
        self._gamma.clear()
        self._blocks.clear()
        if hasattr(self.eris, 'release_blocks'):
            self.eris.release_blocks()

    def memory_report(self):
        '''Bytes of the arrays kept between iterations: [(label, bytes)],
        largest first.  Arrays shared between entries are counted once.'''
        seen = set()

        def size(a):
            base = a
            while getattr(base, 'base', None) is not None:
                base = base.base
            if id(base) in seen:
                return 0
            seen.add(id(base))
            return base.nbytes

        rows = {}

        def add(label, a):
            if a is not None and hasattr(a, 'nbytes'):
                rows[label] = rows.get(label, 0) + size(a)

        if self._vvvv is not None:
            add('(vv|vv) matrix', self._vvvv.g)
        for (bra, ket), (full, staged) in self._compiled.items():
            cons = list(full) + [c for _, cs in staged for c in cs]
            for con in cons:
                kind = type(con).__name__
                tag = '%s<-%s %s' % (bra, ket, kind)
                if kind == '_Contraction':
                    add(tag, getattr(con, 'mat', None) if con.w is None else con.w)
                elif kind == '_TwoStage':
                    add(tag + ' integrals', con.gmat2 if con.g3 is None else con.g3)
                    add(tag + ' Gamma', con.gmat if con.gam3 is None else con.gam3)
                elif kind == '_IntegralFirst':
                    add(tag + ' integrals', getattr(con.first, 'mat', con.first.w))
                    add(tag + ' Gamma', getattr(con.second, 'mat', con.second.w))
                elif kind == '_VVVV':
                    add('three-index (vv) for (vv|vv)', con.b)
        for key, g in self._gamma.items():
            add('Gamma cache %s<-%s' % key[:2], g)
        for (name, labels), a in self._blocks.items():
            add('integral block %s %s' % (name, labels), a)
        for name in ('_cd_pa', '_cd_pc'):
            add('eris ' + name, getattr(self.eris, name, None))
        for labels, a in getattr(self.eris, 'cd', {}).items():
            add('eris three-index %s' % labels, a)
        for labels, a in getattr(self.eris, '_cache', {}).items():
            add('eris block %s' % labels, a)
        return sorted(rows.items(), key=lambda x: -x[1])

    def _tick(self, bra, ket, con, t0):
        '''Time by (bra, ket, kind of contraction): Sigma.timings.'''
        key = bra, ket, (con if isinstance(con, str) else type(con).__name__
                         if con is not None else '')
        self.timings[key] = self.timings.get(key, 0.0) + time.perf_counter() - t0
        # growth of the peak resident size while this contraction ran (GB)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
        if rss > self._rss + 0.05:
            self.peak_growth[key] = self.peak_growth.get(key, 0.0) + rss - self._rss
        self._rss = max(rss, self._rss)

    def matvec(self, x):
        '''H x for packed vectors x (nvec, dim).'''
        x = numpy.atleast_2d(x)
        if len(self._compiled) < len(self.basis.names)**2:
            self.compile_all()
        self._rss = max(self._rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0)
        t0 = time.perf_counter()
        try:
            return self._matvec(x)
        finally:
            self.wall += time.perf_counter() - t0
            self.calls += len(x)

    def _matvec(self, x):
        # one vector at a time: the intermediates of a sigma are not multiplied
        # by the number of vectors
        out = []
        for k, v in enumerate(x):
            start = time.perf_counter()
            before, before_steps = dict(self.timings), dict(self.steps)
            t0 = time.perf_counter()
            c = self.basis.unpack(v[None])
            self._tick('all', 'unpack', None, t0)
            sig = self(c)
            t0 = time.perf_counter()
            out.append(self.basis.pack(sig)[0])
            self._tick('all', 'pack', None, t0)
            seconds = time.perf_counter() - start
            if self.progress is not None and seconds >= self.report_seconds:
                self.progress(self.report(seconds, before, before_steps,
                                          'sigma vector %d' % (self.calls + k + 1)))
        return numpy.stack(out)

    def step_lines(self, key, since=None, indent='      '):
        '''The steps of the contraction key = (bra, ket, kind) as text.'''
        since = since or {}
        rows = [(name, v - since.get((k, name), 0.0)) for (k, name), v in self.steps.items()
                if k == key]
        return ', '.join('%s %.1f' % (name, v) for name, v in sorted(rows, key=lambda r: -r[1])
                         if v >= 0.05)

    def report(self, seconds, before, before_steps, title):
        '''Text: the slowest contractions since the snapshots before / before_steps,
        with their steps.'''
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
        rows = sorted(((v - before.get(k, 0.0), k) for k, v in self.timings.items()),
                      reverse=True)
        lines = ['     %s: %.1f s, peak memory %.1f GB' % (title, seconds, rss)]
        for dt, key in rows[:self.report_lines]:
            if dt < 0.01 * seconds:
                break
            steps = self.step_lines(key, before_steps)
            lines.append('       %-5s <- %-5s %-14s %8.1f s%s' % (
                key[0], key[1], key[2].lstrip('_'), dt, '   (%s)' % steps if steps else ''))
        return '\n'.join(lines)

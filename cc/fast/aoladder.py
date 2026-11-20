'''The AO operators of the pp ladder of DirectZCCSD: kept, or made from the AO
integrals in every iteration (integral-direct).

The ladder contracts the back-transformed amplitudes of the pairs (i>j) with
two real matrices on pairs of scalar AO functions,

    W_same[(mu>nu), (si>rho)]  = (mu si|nu rho) - (mu rho|nu si)
    W_plus[(nu>=si), (mu>=la)] = (mu nu|la si) + (mu si|la nu)

(zccsdt.zccsd.ao2mo.ladder_operators; that W_plus carries a factor 1/2 on its
rows with nu = si, which is put on the amplitudes here).  Both are symmetric,
N^4/4 elements each for N AO functions, and only their lower triangles are
needed.  The rows of a batch of shells of the first index mu (nu in W_plus)
are the rows c0 <= r < c1 of an operator W; with T = W[rows, :c1] for these
rows, or for a part U of them,

    G[:, :c1] += A[:, U] T                the rows against all columns up to c1
    G[:, U]   += A[:, :c0] T[:, :c0]^T    the columns U, for the rows before c0

make every element of G = A W once over all batches, as two GEMMs per part.
The first is a rank-|U| update that reads and writes all of G[:, :c1]: thin
parts cost memory traffic (benzene/cc-pVDZ: 0.68 s for the GEMMs of an
iteration with 250 rows per part, 0.55 s with 800, 0.50 s with the square
matrices in one GEMM each).

The rows come from blocks of AO integrals (f x|t y), "units": f over a batch
of shells, x and y over the functions up to the end of the batch, t over all
of these as well or, when that is too much at once, over a range of shells.
With M[x, y] = (f x|t y), the row (f, t) of W_same is M - M^T on x > y and
that of W_plus M + M^T on x >= y (kernels.ladder_rows).  The triangles need
only the integrals with f the largest index, and libcint computes a unit
with all t with the symmetry (f x|t y) = (f x|y t): N^4/8 integrals for a
pass over both operators, a little more for batches of several functions.

Modes:

'memory'  the rows are kept, in ``nblocks`` parts of about equal numbers of
          rows: one part is the square matrices (2 N^4 bytes each, one GEMM
          each), many parts tend to the triangles (N^4 bytes each).
'direct'  nothing is kept: the integrals are computed in every application
          (once per block of pairs of every iteration), in units of at most
          ``work`` bytes whose rows are multiplied about UNIT_ROWS at a time.
'file'    as 'memory', the parts being mapped arrays of a scratch file that
          are read in order.

Timers (zccsdt.timing): 'ladder.ints' (libcint), 'ladder.rows' (the rows from
the integrals), 'ladder.gemm'; 'ao2mo.ladder' around the rows that are kept.
'''

from dataclasses import dataclass

import numpy as np
from pyscf.gto import moleintor

from socutils.cc.fast import blas, scratch
from socutils.cc.fast.timing import timer
from socutils.cc.fast import kernels as K

UNIT_ROWS = 1024            # rows that are multiplied together (mode 'direct'), about
MAX_BLOCKS = 32             # parts of the kept rows, at most
MODES = ('memory', 'direct', 'file')


def _size(nbytes):
    return '%.2f GB' % (nbytes / 1e9) if nbytes >= 1e9 else '%.1f MB' % (nbytes / 1e6)


def _tri(n, diagonal):
    return n * (n + 1) // 2 if diagonal else n * (n - 1) // 2


def _runs(f0, f1, n0, n1, diagonal):
    '''Runs of consecutive rows (pairs (f, t), f0 <= f < f1, n0 <= t < n1 and
    t < f, or t <= f with ``diagonal``), merged where they touch.'''
    runs = []
    for f in range(f0, f1):
        a = _tri(f, diagonal) + n0
        b = _tri(f, diagonal) + min(n1, f + diagonal)
        if b <= a:
            continue
        if runs and runs[-1][1] == a:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    return runs


def _unit_rows(nao):
    '''Rows of a unit: UNIT_ROWS, fewer for a small basis (thick units compute
    and multiply more of the square of their own rows than the triangle
    needs).'''
    return min(UNIT_ROWS, max(128, _tri(nao, 1) // 16))


@dataclass
class Unit:
    '''A block of AO integrals (f x|t y) and the rows of the operators it
    gives: f over the shells sh0 .. sh1 - 1 (the functions f0 .. f1 - 1), t
    over the shells t0 .. t1 - 1 (the functions n0 .. n1 - 1), x and y over
    the functions below f1.'''
    shells: tuple       # (sh0, sh1, t0, t1)
    f: tuple            # (f0, f1)
    t: tuple            # (n0, n1)
    rows_max: int       # rows that are made and multiplied together, about

    @property
    def m(self):
        return self.f[1]

    @property
    def packed(self):
        '''All functions up to the end of the batch as the third index: the
        integrals come with their symmetry in the last two indices.'''
        return self.t == (0, self.f[1])

    @property
    def nints(self):
        '''Integrals of the block.'''
        nf, nt, m = self.f[1] - self.f[0], self.t[1] - self.t[0], self.m
        return nf * m * (_tri(m, 1) if self.packed else nt * m)

    @property
    def chunks(self):
        '''Ranges (fa, fb) of f whose rows are made and multiplied together:
        about rows_max rows (f + 1 for the function f), one function at
        least; with a range of t, all of them.'''
        if not self.packed:
            return [self.f]
        out, fa, n = [], self.f[0], 0
        for f in range(*self.f):
            if n and n + f + 1 > self.rows_max:
                out.append((fa, f))
                fa, n = f, 0
            n += f + 1
        return out + [(fa, self.f[1])]

    def rows(self, fa, fb):
        '''(rows of W_same, rows of W_plus) of the functions fa .. fb - 1.'''
        return K.ladder_row_counts(fb - fa, fa, self.t[1] - self.t[0], self.t[0])

    def runs(self, fa, fb, diagonal):
        '''The rows of the functions fa .. fb - 1 in W_same, or in W_plus
        (``diagonal``), as runs (a, b) of consecutive rows.'''
        return _runs(fa, fb, self.t[0], self.t[1], diagonal)

    def bounds(self, diagonal):
        '''(c0, c1): the rows of the batch of this unit in W_same, or in
        W_plus (``diagonal``).'''
        return _tri(self.f[0], diagonal), _tri(self.f[1], diagonal)

    @property
    def nbytes(self):
        '''The integrals and the rows of the largest chunk.'''
        m = self.m
        rows = max(ns * _tri(m, 0) + npl * _tri(m, 1)
                   for ns, npl in (self.rows(fa, fb) for fa, fb in self.chunks))
        return 8 * (self.nints + rows)


def unit_floor(mol):
    '''Bytes of the largest unit of one shell against one shell: the least
    the units can be made of.'''
    loc = mol.ao_loc_nr()
    nsh = int((loc[1:] - loc[:-1]).max())
    nao = int(loc[-1])
    return 16 * nsh * nsh * nao * (nao + 1)


def units(mol, work=None, shells=None):
    '''The Units for the shells ``shells`` = (s0, s1) of the first index
    (default: all), with at most ``work`` bytes each (None: no limit).  A
    unit is a batch of shells with all functions below its end as the third
    index; it grows to the rows of one chunk.  Where one shell is too much
    for ``work``, it comes with ranges of shells of the third index (and
    without the symmetry of the integrals in the last two indices).'''
    loc = [int(x) for x in mol.ao_loc_nr()]
    nao = loc[-1]
    rows_max = _unit_rows(nao)
    s0, s1 = (0, mol.nbas) if shells is None else shells

    def packed(sh0, sh1):
        return Unit((sh0, sh1, 0, sh1), (loc[sh0], loc[sh1]), (0, loc[sh1]), rows_max)

    def fits(u):
        return work is None or u.nbytes <= work

    out, sh0 = [], s0
    while sh0 < s1:
        sh1 = sh0 + 1
        if fits(packed(sh0, sh1)):
            while (sh1 < s1 and _tri(loc[sh1 + 1], 1) - _tri(loc[sh0], 1) <= rows_max
                   and fits(packed(sh0, sh1 + 1))):
                sh1 += 1
            out.append(packed(sh0, sh1))
        else:
            t0 = 0
            while t0 < sh1:
                t1 = t0 + 1
                u = Unit((sh0, sh1, t0, t1), (loc[sh0], loc[sh1]), (loc[t0], loc[t1]), rows_max)
                if not fits(u):
                    raise MemoryError(
                        'the AO integrals of the ladder need %s for one shell against one '
                        'shell (%d AO functions), %s are there'
                        % (_size(u.nbytes), nao, _size(work)))
                while t1 < sh1:
                    bigger = Unit((sh0, sh1, t0, t1 + 1), u.f, (loc[t0], loc[t1 + 1]), rows_max)
                    if not fits(bigger):
                        break
                    u, t1 = bigger, t1 + 1
                out.append(u)
                t0 = t1
        sh0 = sh1
    return out


def unit_bytes(mol):
    '''Bytes of the largest unit when nothing limits the units.'''
    return max(u.nbytes for u in units(mol))


def parts(mol, nblocks):
    '''Ranges of shells (s0, s1) of the first index for ``nblocks`` parts of
    the rows of about equal size (fewer if there are not that many shells).'''
    loc = [int(x) for x in mol.ao_loc_nr()]
    total, out, s0 = _tri(loc[-1], 1), [], 0
    for k in range(1, nblocks + 1):
        s1 = s0
        while s1 < mol.nbas and (k == nblocks or _tri(loc[s1 + 1], 1) * nblocks <= k * total):
            s1 += 1
        if s1 > s0:
            out.append((s0, s1))
            s0 = s1
    return out


def stored_bytes(mol, nblocks=1):
    '''Bytes of both operators when their rows are kept in ``nblocks``
    parts: the square matrices for one part, towards half of that for many.'''
    loc = mol.ao_loc_nr()
    return 8 * sum((_tri(int(loc[s1]), d) - _tri(int(loc[s0]), d)) * _tri(int(loc[s1]), d)
                   for s0, s1 in parts(mol, nblocks) for d in (0, 1))


def most_blocks(mol):
    '''The largest number of parts that is useful: the parts are not to be
    thinner than UNIT_ROWS rows.'''
    return max(1, min(MAX_BLOCKS, _tri(mol.nao_nr(), 1) // UNIT_ROWS))


def blocks_that_fit(mol, room):
    '''The smallest number of parts with which the kept operators take no
    more than ``room`` bytes (None: no limit); None if they never do.'''
    nb = 1
    while True:
        if room is None or stored_bytes(mol, nb) <= room:
            return nb
        if nb >= most_blocks(mol):
            return None
        nb = min(2 * nb, most_blocks(mol))


class AOLadder:
    '''The AO operators of the ladder for ``mol`` in one of the MODES; see the
    module.  ``work``: bytes of a unit of integrals at most (None: no
    limit).  ``nblocks``: parts of the rows that are kept.  ``auto``: the
    mode was not asked for, and rows kept in memory may give way to the mode
    'direct' when the step does not fit next to them (to_direct).'''

    def __init__(self, mol, mode='memory', work=None, nblocks=1, auto=False):
        assert mode in MODES
        self.mol, self.mode, self.nblocks, self.auto = mol, mode, nblocks, auto
        self.nao = mol.nao_nr()
        self.intor = mol._add_suffix('int2e')
        self._opt = moleintor.make_cintopt(mol._atm, mol._bas, mol._env, self.intor)
        self._loc = mol.ao_loc_nr()
        self.diagonal = np.flatnonzero(np.equal(*np.tril_indices(self.nao)))    # nu = si in W_plus
        self.same = self.plus = None        # kept rows: lists of (runs, T, c0)
        self.scratch = None
        self._buf = None
        self.passes = 0                     # applications so far
        self.set_work(work)
        self.keep()

    # -- units ---------------------------------------------------------------
    def set_work(self, work):
        '''Units of at most ``work`` bytes (None: no limit), for the
        integrals that are still to be computed: every application in the
        mode 'direct'.'''
        self._work = None if work is None else max(int(work), unit_floor(self.mol))
        self.units = units(self.mol, self._work)
        self.work = max(u.nbytes for u in self.units)
        self._buf = None

    def _buffers(self, us):
        '''The arrays of the largest of the units ``us``: integrals, rows of
        W_same, rows of W_plus.  They are kept from one unit to the next (new
        arrays are mapped and zeroed page by page) until release().'''
        if self._buf is None:
            ne = max(u.nints for u in us)
            ns = max(u.rows(*c)[0] * _tri(u.m, 0) for u in us for c in u.chunks)
            npl = max(u.rows(*c)[1] * _tri(u.m, 1) for u in us for c in u.chunks)
            self._buf = tuple(np.empty(max(n, 1)) for n in (ne, ns, npl))
        return self._buf

    def blocks(self, u, bufs):
        '''The rows of the unit ``u``, chunk by chunk: for W_same and for
        W_plus (runs, T, c0) with T[k, c] = W[r_k, c], c < c1, for the rows
        r_k of the runs (views of the arrays ``bufs``: valid until the next
        chunk).'''
        bufe, bufs, bufp = bufs
        sh0, sh1, t0, t1 = u.shells
        (f0, f1), (n0, n1), m = u.f, u.t, u.m
        with timer('ladder.ints'):
            E = moleintor.getints4c(self.intor, self.mol._atm, self.mol._bas, self.mol._env,
                                    shls_slice=(sh0, sh1, 0, sh1, t0, t1, 0, sh1),
                                    aosym='s2kl' if u.packed else 's1',
                                    ao_loc=self._loc, cintopt=self._opt, out=bufe)
            E = E.reshape((f1 - f0, m, -1) if u.packed else (f1 - f0, m, n1 - n0, m))
        ws, wp = _tri(m, 0), _tri(m, 1)
        for fa, fb in u.chunks:
            with timer('ladder.rows'):
                ns, npl = u.rows(fa, fb)
                S = bufs[:ns * ws].reshape(ns, ws) if ns else np.empty((0, max(ws, 1)))
                P = bufp[:npl * wp].reshape(npl, wp)
                K.ladder_rows(E[fa - f0:fb - f0], fa, n0, S, P)
            yield ((u.runs(fa, fb, 0), S, u.bounds(0)[0]),
                   (u.runs(fa, fb, 1), P, u.bounds(1)[0]))

    def keep(self):
        '''Compute all rows once and keep them, in nblocks parts (in memory,
        or in a scratch file).  Nothing happens in the mode 'direct', or when
        they are there.'''
        if self.mode == 'direct' or self.same is not None:
            return
        if self.mode == 'file':
            self.scratch = scratch.Scratch(
                prefix='zccsd_w_', what='the AO operators of the ladder',
                hint=' or let them be computed in every iteration (ao_ladder = \'direct\')')
        loc = [int(x) for x in self._loc]
        self.same, self.plus = [], []
        with timer('ao2mo.ladder'):
            for k, (s0, s1) in enumerate(parts(self.mol, self.nblocks)):
                f0, f1 = loc[s0], loc[s1]
                us = units(self.mol, self._work, (s0, s1))
                self._buf = None
                bufs = self._buffers(us)
                part = {}
                for d, name, kept in ((0, 's', self.same), (1, 'p', self.plus)):
                    a0, a1 = _tri(f0, d), _tri(f1, d)
                    if a1 == a0:
                        continue
                    if self.mode == 'file':
                        T = self.scratch.create('%s%d' % (name, k), (a1 - a0, a1), np.float64)
                    else:
                        T = np.empty((a1 - a0, a1))
                    part[d] = T
                    kept.append(([(a0, a1)], T, a0))
                for u in us:
                    for both in self.blocks(u, bufs):
                        for d, (runs, R, _) in enumerate(both):
                            a0, off = _tri(f0, d), 0
                            for a, b in runs:
                                part[d][a - a0:b - a0, :R.shape[1]] = R[off:off + b - a]
                                off += b - a
                # the rows of a batch of shells end at its own last column: the
                # rest of the square of the part is the transpose of the rows below
                for d, T in part.items():
                    a0 = _tri(f0, d)
                    D = T[:, a0:]
                    for g0, g1 in sorted({u.f for u in us})[:-1]:
                        a, b = _tri(g0, d) - a0, _tri(g1, d) - a0
                        D[a:b, b:] = D[b:, a:b].T
                    if self.mode == 'file':
                        scratch.written(T)
                        scratch.unmap(T)
        self._buf = None

    def release(self):
        '''Free the arrays of the units (they are kept from one application
        to the next).'''
        self._buf = None

    def unit_range(self):
        '''(least, most) bytes of a unit that is computed or read in every
        application, for the memory plan of the step; None in the mode
        'memory'.'''
        if self.mode == 'memory':
            return None
        if self.mode == 'file':     # the largest part is mapped while it is read
            loc = self._loc
            big = 8 * max((_tri(int(loc[s1]), 1) - _tri(int(loc[s0]), 1)) * _tri(int(loc[s1]), 1)
                          for s0, s1 in parts(self.mol, self.nblocks))
            return big, big
        return unit_floor(self.mol), unit_bytes(self.mol)

    def close(self):
        '''Drop what is kept (keep() makes it again).'''
        self.same = self.plus = self._buf = None
        if self.scratch is not None:
            self.scratch.close()
            self.scratch = None

    def to_direct(self):
        '''Drop what is kept and compute the integrals in every application
        from now on.'''
        self.close()
        self.mode = 'direct'

    # -- the operators -----------------------------------------------------------
    @staticmethod
    def _rows_times(A, G, runs, T, c0):
        '''G += A W for the rows of ``runs`` of the symmetric W, T[k] = W[r_k,
        :c1], and for the columns of these rows against the rows before c0.'''
        if not runs:
            return
        c1 = T.shape[1]
        if len(runs) == 1:
            a, b = runs[0]
            blas.gemm(A[:, a:b], T, C=G[:, :c1], beta=1.0)
            if c0:
                blas.gemm(A[:, :c0], T[:, :c0].T, C=G[:, a:b], beta=1.0)
            return
        blas.gemm(np.concatenate([A[:, a:b] for a, b in runs], axis=1), T, C=G[:, :c1], beta=1.0)
        if c0:
            X = blas.gemm(A[:, :c0], T[:, :c0].T)
            off = 0
            for a, b in runs:
                G[:, a:b] += X[:, off:off + b - a]
                off += b - a

    def apply(self, AR, SR):
        '''(AR W_same, SR W_plus) for the rows AR[k, (mu>nu)] and
        SR[k, (nu>=si)] of kernels.ladder_pack.  SR is overwritten.'''
        SR[:, self.diagonal] *= 0.5
        self.passes += 1
        self.keep()
        if self.mode == 'memory' and len(self.plus) == 1:      # the square matrices
            with timer('ladder.gemm'):
                GS = blas.gemm(SR, self.plus[0][1])
                GA = blas.gemm(AR, self.same[0][1]) if self.same else np.zeros_like(AR)
            return GA, GS
        GA, GS = np.zeros_like(AR), np.zeros_like(SR)
        if self.mode == 'direct':
            bufs = self._buffers(self.units)
            for u in self.units:
                for same, plus in self.blocks(u, bufs):
                    with timer('ladder.gemm'):
                        self._rows_times(AR, GA, *same)
                        self._rows_times(SR, GS, *plus)
            return GA, GS
        with timer('ladder.gemm'):
            for A, G, kept in ((AR, GA, self.same), (SR, GS, self.plus)):
                for runs, T, c0 in kept:
                    if self.mode == 'file':
                        scratch.advise(T, 'sequential', force=True)
                    self._rows_times(A, G, runs, T, c0)
                    scratch.unmap(T)
        return GA, GS

    # -- what there is ---------------------------------------------------------
    @property
    def nbytes(self):
        '''Bytes of the rows that are kept.'''
        if self.same is None:
            return 0
        return sum(T.nbytes for _, T, _ in self.same + self.plus)

    def describe(self):
        n = self.nao
        if self.mode == 'direct':
            return ('computed in every iteration from the AO integrals (%.2e integrals in %d '
                    'units of at most %s)' % (sum(u.nints for u in self.units),
                                               len(self.units), _size(self.work)))
        where = 'in memory' if self.mode == 'memory' else 'in a scratch file'
        if self.same is None:
            return '%s (dropped)' % where
        return ('%s: W_same and W_plus (%d x %d and %d x %d), %s in %d part%s of their rows'
                % (where, _tri(n, 0), _tri(n, 0), _tri(n, 1), _tri(n, 1), _size(self.nbytes),
                   len(self.plus), '' if len(self.plus) == 1 else 's'))

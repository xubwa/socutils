#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Full and generalized-active-space (GAS, RAS) CI for spinor Hamiltonians, and
for non-relativistic ones over spatial orbitals: driver of the C kernels in
socutils/lib/zgasci (lib/libzgasci.so).

The division of labour is that of PySCF's FCI: the kernels apply the
Hamiltonian to a CI vector and build density matrices; the vectors live in
Python and are diagonalized with pyscf.lib.davidson1.  GASCI has the
interface and the density-matrix conventions of socutils.hci.shci.SHCI and
socutils.fci.zfci.FCISolver, so it can stand in for them as the fcisolver of
socutils.mcscf.zmcscf.CASSCF::

    mc = zmcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver = gasci.GASCI(mol)
    mc.fcisolver.gas = [dict(orbitals=range(4), min=3, max=4),
                        dict(orbitals=range(4, ncas), min=0, max=1)]
    mc.kernel()

SpatialGASCI is the counterpart over spatial orbitals, with the interface of
pyscf.fci.direct_spin1 (and of SpatialSHCI); it is a fcisolver for
pyscf.mcscf.CASSCF.

Full CI is the case without constraints, RAS a special GAS.  The constraints
(``gas`` and ``restrictions``) bound the number of electrons in sets of
orbitals.  The spinors are reordered so that every such set is a union of
consecutive sub-groups of at most ``max_subgroup`` spinors, half of every set
on each of the two sides A and B of the determinant strings (see
lib/zgasci/zgasci.h); ``subgroups`` gives that division explicitly.  None of
this changes the result, only how fast it comes.

CI vectors are numpy arrays in the order of the determinants of the plan
(``plan(norb, nelec).occslst()``); ``wavefunction`` converts them to
determinants of the original orbitals in increasing order.
'''

import ctypes
import itertools

import numpy

from pyscf import lib
from pyscf.lib import logger

from socutils.lib import load_library

_lib = None


def _load():
    global _lib
    if _lib is None:
        lib_ = load_library('libzgasci', 'SOCUTILS_ZGASCI_LIBRARY')
        ip = ctypes.POINTER(ctypes.c_int)
        lib_.zg_plan_new.restype = ctypes.c_void_p
        lib_.zg_plan_new.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ip, ctypes.c_int, ip, ip, ip,
                                     ip, ip, ctypes.c_int, ip, ip, ip]
        lib_.zg_plan_free.argtypes = [ctypes.c_void_p]
        lib_.zg_error.restype = ctypes.c_char_p
        lib_.zg_dim.restype = ctypes.c_int64
        lib_.zg_dim.argtypes = [ctypes.c_void_p]
        lib_.zg_nwords.restype = ctypes.c_int
        lib_.zg_nwords.argtypes = [ctypes.c_void_p]
        for name in ('zg_info', 'zg_dets'):
            getattr(lib_, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        for name in ('zg_hdiag_d', 'zg_hdiag_z'):
            getattr(lib_, name).argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 2
        lib_.zg_block_ok.restype = ctypes.c_int
        lib_.zg_block_ok.argtypes = [ctypes.c_void_p]
        for name in ('zg_sigma_d', 'zg_sigma_z', 'zg_rdm2_d', 'zg_rdm2_z',
                     'zg_sigma_block_d', 'zg_sigma_block_z',
                     'zg_sigma_blas_d', 'zg_sigma_blas_z',
                     'zg_rdm2_blas_d', 'zg_rdm2_blas_z', 'zg_det_strings'):
            getattr(lib_, name).argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 3
        lib_.zg_det_strings.argtypes = [ctypes.c_void_p] * 3
        lib_.zg_plan_product.restype = ctypes.c_void_p
        lib_.zg_plan_product.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                         ctypes.c_int64, ctypes.c_void_p,
                                         ctypes.c_int64, ctypes.c_void_p,
                                         ctypes.c_int, ip, ip, ip]
        lib_.zg_is_product.restype = ctypes.c_int
        lib_.zg_is_product.argtypes = [ctypes.c_void_p]
        lib_.zg_is_mirror.restype = ctypes.c_int
        lib_.zg_is_mirror.argtypes = [ctypes.c_void_p]
        for name in ('zg_flip_vector_d', 'zg_flip_vector_z'):
            getattr(lib_, name).restype = ctypes.c_int
            getattr(lib_, name).argtypes = [ctypes.c_void_p] * 3
        for name in ('zg_sigma_flip_d', 'zg_sigma_flip_z'):
            getattr(lib_, name).restype = ctypes.c_int
            getattr(lib_, name).argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int]
        for name in ('zg_transpose_d', 'zg_copy_block_d'):
            getattr(lib_, name).argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
                                            ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64]
        for name in ('zg_sigma_ab_rows_d', 'zg_sigma_ab_rows_z'):
            getattr(lib_, name).restype = ctypes.c_int
            getattr(lib_, name).argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 2
        lib_.zg_product_strings.restype = ctypes.c_int64
        lib_.zg_product_strings.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib_.zg_product_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _lib = lib_
    return _lib


def _intp(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def pair_indices(n):
    '''p[P], q[P] of the pairs P = q (q - 1) / 2 + p, p < q.'''
    q, p = numpy.tril_indices(n, -1)
    return p, q


def antisymmetrized(h1, eri, n, nelec):
    '''v[P,R] = <pq||rs> + the one-electron part folded in, P = (p<q),
    R = (r<s): the Hamiltonian sum_{P,R} v[P,R] a+_p a+_q a_s a_r on states
    of nelec electrons.  eri[p,q,r,s] = (pq|rs).'''
    eri = numpy.asarray(eri).reshape(n, n, n, n)
    g = eri.transpose(0, 2, 1, 3)  # <pq|rs> = (pr|qs)
    w = g - g.transpose(0, 1, 3, 2)
    d = numpy.eye(n)
    hd = numpy.einsum('pr,qs->pqrs', h1, d)
    w = w + (hd - hd.transpose(0, 1, 3, 2) - hd.transpose(1, 0, 2, 3)
             + hd.transpose(1, 0, 3, 2)) / (nelec - 1)
    p, q = pair_indices(n)
    return numpy.ascontiguousarray(w[p[:, None], q[:, None], p[None, :], q[None, :]])


def s2_antisymmetrized(norb, nalpha, nbeta):
    '''v[P,R] of S^2 over the spin orbitals of a spatial plan (alpha P = P,
    beta P = norb + P) on states of nalpha + nbeta electrons, in the format of
    antisymmetrized: S^2 = N_beta + S_z (S_z + 1)
    - sum_pq a+_{p alpha} a+_{q beta} a_{p beta} a_{q alpha}.  No integrals:
    the coefficients are -1 and the constant.'''
    n = 2 * norb
    nelec = nalpha + nbeta
    sz = 0.5 * (nalpha - nbeta)
    w = numpy.zeros((n, n, n, n))
    p = numpy.arange(norb)
    # w[x, y, r, s]: coefficient of a+_x a+_y a_s a_r
    w[p[:, None], norb + p[None, :], p[None, :], norb + p[:, None]] = -1
    w = w - w.transpose(1, 0, 2, 3) - w.transpose(0, 1, 3, 2) + w.transpose(1, 0, 3, 2)
    h = numpy.eye(n) * ((nbeta + sz * (sz + 1)) / nelec)
    hd = numpy.einsum('pr,qs->pqrs', h, numpy.eye(n))
    w += (hd - hd.transpose(0, 1, 3, 2) - hd.transpose(1, 0, 2, 3)
          + hd.transpose(1, 0, 3, 2)) / (nelec - 1)
    a, b = pair_indices(n)
    return numpy.ascontiguousarray(w[a[:, None], b[:, None], a[None, :], b[None, :]])


def unpack_rdm2(g, n, nelec):
    '''dm1[p,r] = <p^+ r>, dm2[p,r,q,s] = <p^+ q^+ s r> from
    g[P,R] = <a+_p a+_q a_s a_r>.'''
    p, q = pair_indices(n)
    P, Q, R, S = p[:, None], q[:, None], p[None, :], q[None, :]
    dm2 = numpy.zeros((n, n, n, n), dtype=g.dtype)
    dm2[P, R, Q, S] = g
    dm2[Q, R, P, S] = -g
    dm2[P, S, Q, R] = -g
    dm2[Q, S, P, R] = g
    dm1 = numpy.einsum('prqq->pr', dm2) / (nelec - 1)
    return dm1, dm2


class Plan:
    '''The addressing scheme of a CI space over n spinors in the order of the
    kernels (lib/zgasci/zgasci.h).

    sub_sizes: spinors of each sub-group, the first nsub_a on side A;
    constraints: list of (sub-groups, min, max) or (sub-groups, min, max,
    weights), bounding sum weight * electrons over the sub-groups (weights 1
    if not given); labels: (nsets, n) integer array, mods and targets one
    per set.
    '''

    def __init__(self, n, nelec, sub_sizes, nsub_a, constraints=(),
                 labels=None, mods=(), targets=()):
        self._lib = _load()
        self.n, self.nelec = int(n), int(nelec)
        self.npair = self.n * (self.n - 1) // 2
        sub_sizes = numpy.asarray(sub_sizes, dtype=numpy.intc)
        gas_ptr = numpy.zeros(len(constraints) + 1, dtype=numpy.intc)
        gas_sub, gas_wt, gas_min, gas_max = [], [], [], []
        for k, item in enumerate(constraints):
            subs, lo, hi = item[:3]
            weights = item[3] if len(item) > 3 else [1] * len(subs)
            gas_sub.extend(int(x) for x in subs)
            gas_wt.extend(int(x) for x in weights)
            gas_ptr[k + 1] = len(gas_sub)
            gas_min.append(int(lo))
            gas_max.append(int(hi))
        gas_sub = numpy.asarray(gas_sub + [0], dtype=numpy.intc)
        gas_wt = numpy.asarray(gas_wt + [0], dtype=numpy.intc)
        gas_min = numpy.asarray(gas_min + [0], dtype=numpy.intc)
        gas_max = numpy.asarray(gas_max + [0], dtype=numpy.intc)
        if labels is None:
            labels = numpy.zeros((0, self.n), dtype=numpy.intc)
        labels = numpy.ascontiguousarray(numpy.reshape(labels, (-1, self.n)),
                                         dtype=numpy.intc)
        nsets = labels.shape[0]
        mods = numpy.asarray(list(mods) + [0], dtype=numpy.intc)
        targets = numpy.asarray(list(targets) + [0], dtype=numpy.intc)
        self._keep = (sub_sizes, gas_ptr, gas_sub, gas_wt, gas_min, gas_max,
                      labels, mods, targets)
        handle = self._lib.zg_plan_new(
            self.n, self.nelec, len(sub_sizes), int(nsub_a), _intp(sub_sizes),
            len(constraints), _intp(gas_ptr), _intp(gas_sub), _intp(gas_wt),
            _intp(gas_min),
            _intp(gas_max), nsets, _intp(labels.reshape(-1) if nsets else
                                         numpy.zeros(1, dtype=numpy.intc)),
            _intp(mods), _intp(targets))
        if not handle:
            raise ValueError('zgasci: ' + self._lib.zg_error().decode())
        self._handle = ctypes.c_void_p(handle)
        self.sub_sizes = [int(x) for x in sub_sizes]
        self.nsub_a = int(nsub_a)
        self.dim = int(self._lib.zg_dim(self._handle))
        self.nwords = int(self._lib.zg_nwords(self._handle))
        self.block_ok = bool(self._lib.zg_block_ok(self._handle))

    def __del__(self):
        handle = getattr(self, '_handle', None)
        if handle is not None and handle.value:
            self._lib.zg_plan_free(handle)
            self._handle = None

    def info(self):
        out = numpy.zeros(8, dtype=numpy.int64)
        self._lib.zg_info(self._handle, out.ctypes.data)
        keys = ('strings_a', 'strings_b', 'classes_a', 'classes_b',
                'buckets_a', 'buckets_b', 'blocks', 'tasks')
        return dict(zip(keys, (int(x) for x in out)))

    def dets(self):
        '''words[dim, nwords]: bit p set if spinor p is occupied.'''
        words = numpy.zeros((self.dim, self.nwords), dtype=numpy.uint64)
        self._lib.zg_dets(self._handle, words.ctypes.data)
        return words

    def occslst(self):
        '''The occupied spinors of every determinant, (dim, nelec).'''
        words = self.dets()
        bits = numpy.zeros((self.dim, self.n), dtype=bool)
        for p in range(self.n):
            bits[:, p] = (words[:, p // 64] >> numpy.uint64(p % 64)) & numpy.uint64(1)
        return numpy.nonzero(bits)[1].reshape(self.dim, self.nelec)

    def _check_v(self, v):
        v = numpy.asarray(v)
        if v.shape != (self.npair, self.npair):
            raise ValueError('v must be (npair, npair)')
        if numpy.iscomplexobj(v):
            return numpy.ascontiguousarray(v, dtype=numpy.complex128), True
        return numpy.ascontiguousarray(v, dtype=numpy.float64), False

    def _vector(self, c, complex_):
        c = numpy.ascontiguousarray(c, dtype=numpy.complex128 if complex_ else
                                    numpy.float64)
        if c.size != self.dim:
            raise ValueError('CI vector of size %d for a space of %d'
                             % (c.size, self.dim))
        return c

    def hdiag(self, v):
        v, complex_ = self._check_v(v)
        out = numpy.empty(self.dim)
        fn = self._lib.zg_hdiag_z if complex_ else self._lib.zg_hdiag_d
        fn(self._handle, v.ctypes.data, out.ctypes.data)
        return out

    def sigma(self, v, c, kernel='blas'):
        '''H c.  kernel: 'blas' (default; gather - matrix product - scatter,
        the fastest), 'block' or 'reference'.  'blas' and 'block' write each
        row of the result by one thread.  'blas' uses two more vectors of
        memory while it runs.'''
        v, complex_ = self._check_v(v)
        complex_ = complex_ or numpy.iscomplexobj(c)
        if complex_:
            v = v.astype(numpy.complex128)
        c = self._vector(c, complex_)
        out = numpy.empty_like(c)
        if kernel not in ('blas', 'block', 'reference'):
            raise ValueError("kernel must be 'blas', 'block' or 'reference'")
        if kernel == 'blas' and self.block_ok:
            fn = self._lib.zg_sigma_blas_z if complex_ else self._lib.zg_sigma_blas_d
        elif kernel == 'block' and self.block_ok:
            fn = self._lib.zg_sigma_block_z if complex_ else self._lib.zg_sigma_block_d
        else:
            fn = self._lib.zg_sigma_z if complex_ else self._lib.zg_sigma_d
        fn(self._handle, v.ctypes.data, c.ctypes.data, out.ctypes.data)
        return out

    def rdm2_pairs(self, bra, ket, kernel='reference'):
        '''g[P,R] = <bra| a+_p a+_q a_s a_r |ket>.  kernel: 'reference' or
        'blas' (the loops of the BLAS kernel; product plans always).'''
        complex_ = numpy.iscomplexobj(bra) or numpy.iscomplexobj(ket)
        bra, ket = self._vector(bra, complex_), self._vector(ket, complex_)
        out = numpy.empty((self.npair, self.npair), dtype=bra.dtype)
        if kernel == 'blas':
            fn = self._lib.zg_rdm2_blas_z if complex_ else self._lib.zg_rdm2_blas_d
        else:
            fn = self._lib.zg_rdm2_z if complex_ else self._lib.zg_rdm2_d
        fn(self._handle, bra.ctypes.data, ket.ctypes.data, out.ctypes.data)
        return out

    def det_strings(self):
        '''ia[dim], ib[dim]: the A and the B string of each determinant
        (numbers within each side).'''
        ia = numpy.empty(self.dim, dtype=numpy.int64)
        ib = numpy.empty(self.dim, dtype=numpy.int64)
        self._lib.zg_det_strings(self._handle, ia.ctypes.data, ib.ctypes.data)
        return ia, ib


def string_words(strs, nbits):
    '''Strings as 64-bit words, (len(strs), ceil(nbits / 64)) uint64: from a
    sequence of integers (bit j: orbital j occupied) or a 2-D array of
    occupations (len(strs), nbits) of 0 and 1.'''
    nw = (nbits + 63) // 64
    if isinstance(strs, numpy.ndarray) and strs.ndim == 2:
        if strs.shape[1] != nbits:
            raise ValueError('occupation arrays need %d columns' % nbits)
        occ = numpy.asarray(strs != 0)
        out = numpy.zeros((len(occ), nw), dtype=numpy.uint64)
        for j in range(nbits):
            out[occ[:, j], j // 64] |= numpy.uint64(1) << numpy.uint64(j % 64)
        return out
    if isinstance(strs, numpy.ndarray) and strs.dtype.kind in 'iu' and nw == 1:
        if strs.dtype.kind == 'i' and (strs < 0).any():
            raise ValueError('negative string')
        return strs.astype(numpy.uint64).reshape(-1, 1)
    out = numpy.zeros((len(strs), nw), dtype=numpy.uint64)
    mask = (1 << 64) - 1
    for i, x in enumerate(strs):
        x = int(x)
        if x < 0 or x >> nbits:
            raise ValueError('string %d does not fit in %d bits' % (x, nbits))
        for j in range(nw):
            out[i, j] = (x >> (64 * j)) & mask
    return out


def words_to_ints(words):
    '''The strings of string_words as Python integers (a uint64 array if they
    are one word each).'''
    words = numpy.asarray(words, dtype=numpy.uint64)
    if words.shape[1] == 1:
        return words[:, 0].copy()
    return [sum(int(w) << (64 * j) for j, w in enumerate(row)) for row in words]


class ProductPlan(Plan):
    '''The plan of a product space (lib/zgasci/PRODUCT_SPACE.md): every A
    string of words_a times every B string of words_b with nelec electrons
    together and the labels adding up to the targets.  Side A holds the
    spinors 0 .. nside_a - 1, side B the others; words_x as string_words over
    the spinors of the side (bit j: spinor j of the side).  The strings are
    renumbered (electron count, labels, value): strings(side).'''

    def __init__(self, n, nelec, nside_a, words_a, words_b, labels=None,
                 mods=(), targets=()):
        self._lib = _load()
        self.n, self.nelec, self.nside_a = int(n), int(nelec), int(nside_a)
        self.npair = self.n * (self.n - 1) // 2
        nw = [(self.nside_a + 63) // 64, (self.n - self.nside_a + 63) // 64]
        words = []
        for side, w in enumerate((words_a, words_b)):
            w = numpy.ascontiguousarray(w, dtype=numpy.uint64)
            if w.ndim != 2 or w.shape[1] != nw[side]:
                raise ValueError('strings of side %s need %d words each'
                                 % ('AB'[side], nw[side]))
            words.append(w)
        if labels is None:
            labels = numpy.zeros((0, self.n), dtype=numpy.intc)
        labels = numpy.ascontiguousarray(numpy.reshape(labels, (-1, self.n)),
                                         dtype=numpy.intc)
        nsets = labels.shape[0]
        mods = numpy.asarray(list(mods) + [0], dtype=numpy.intc)
        targets = numpy.asarray(list(targets) + [0], dtype=numpy.intc)
        flat = labels.reshape(-1) if nsets else numpy.zeros(1, dtype=numpy.intc)
        self._keep = (words, flat, mods, targets)
        handle = self._lib.zg_plan_product(
            self.n, self.nelec, self.nside_a, len(words[0]), words[0].ctypes.data,
            len(words[1]), words[1].ctypes.data, nsets, _intp(flat), _intp(mods),
            _intp(targets))
        if not handle:
            raise ValueError('zgasci: ' + self._lib.zg_error().decode())
        self._handle = ctypes.c_void_p(handle)
        self._keep = None
        self.dim = int(self._lib.zg_dim(self._handle))
        self.nwords = int(self._lib.zg_nwords(self._handle))
        self.block_ok = True
        self.side_words = nw

    def info(self):
        out = super().info()
        extra = numpy.zeros(4, dtype=numpy.int64)
        self._lib.zg_product_info(self._handle, extra.ctypes.data)
        out.pop('tasks')
        out['table_entries_a'], out['table_entries_b'] = int(extra[2]), int(extra[3])
        return out

    def strings(self, side):
        '''The strings of side 0 (A) or 1 (B) in the order of the plan, as
        words (n, nwords).'''
        count = self._lib.zg_product_strings(self._handle, side, None)
        out = numpy.zeros((count, self.side_words[side]), dtype=numpy.uint64)
        self._lib.zg_product_strings(self._handle, side, out.ctypes.data)
        return out

    def sigma(self, v, c, kernel='blas'):
        '''H c (the BLAS kernel whatever kernel says).'''
        return super().sigma(v, c, kernel='blas')

    def rdm2_pairs(self, bra, ket, kernel='blas'):
        return super().rdm2_pairs(bra, ket, kernel='blas')

    @property
    def mirror(self):
        '''Whether both sides have the same strings and labels.'''
        return bool(self._lib.zg_is_mirror(self._handle))

    def flip_index(self):
        '''For each determinant (a, b) of a mirror plan, the place of (b, a).'''
        ia, ib = self.det_strings()
        n = len(self.strings(1))
        key = ia * n + ib
        order = numpy.argsort(key, kind='stable')
        return order[numpy.searchsorted(key[order], ib * n + ia)]

    def flip_vector(self, c):
        '''C'[a, b] = C[b, a] on a mirror plan (zg_flip_vector).'''
        complex_ = numpy.iscomplexobj(c)
        c = self._vector(c, complex_)
        out = numpy.empty_like(c)
        fn = self._lib.zg_flip_vector_z if complex_ else self._lib.zg_flip_vector_d
        if not fn(self._handle, c.ctypes.data, out.ctypes.data):
            raise ValueError('flip_vector needs a mirror plan')
        return out

    def sigma_flip(self, v, c, parity):
        '''H c for c with C[b, a] = parity C[a, b] (zg_sigma_flip): without
        the B-string half of the kernel.  None if the plan or the integrals
        do not qualify (see zgasci.h).'''
        v, complex_ = self._check_v(v)
        complex_ = complex_ or numpy.iscomplexobj(c)
        if complex_:
            v = v.astype(numpy.complex128)
        c = self._vector(c, complex_)
        out = numpy.empty_like(c)
        fn = self._lib.zg_sigma_flip_z if complex_ else self._lib.zg_sigma_flip_d
        if not fn(self._handle, v.ctypes.data, c.ctypes.data, out.ctypes.data, int(parity)):
            return None
        return out

    def sigma_ab_rows(self, v, c, smap, sig, t_lo, t_hi):
        '''sig += the (AB,AB) terms into the target rows t_lo .. t_hi - 1
        (sig, rows of all B strings) from the source rows held in c: row s of
        the plan is row smap[s] of c where smap[s] >= 0 (zg_sigma_ab_rows;
        product plans with one bucket per side).  Rows are A strings in the
        order of the plan.  False if the plan does not qualify.'''
        v, complex_ = self._check_v(v)
        complex_ = complex_ or numpy.iscomplexobj(c) or numpy.iscomplexobj(sig)
        dtype = numpy.complex128 if complex_ else numpy.float64
        if complex_:
            v = v.astype(numpy.complex128)
        c = numpy.ascontiguousarray(c, dtype=dtype)
        smap = numpy.ascontiguousarray(smap, dtype=numpy.int64)
        if sig.dtype != dtype or not sig.flags.c_contiguous:
            raise ValueError('sig must be a C-contiguous array of %s' % dtype)
        na, nb = self.side_count(0), self.side_count(1)
        if smap.size != na or sig.size != (t_hi - t_lo) * nb or c.size % nb:
            raise ValueError('smap, c or sig of the wrong size')
        if smap.max(initial=-1) >= c.size // nb:
            raise ValueError('smap points past the rows of c')
        fn = self._lib.zg_sigma_ab_rows_z if complex_ else self._lib.zg_sigma_ab_rows_d
        return bool(fn(self._handle, v.ctypes.data, c.ctypes.data, smap.ctypes.data,
                       sig.ctypes.data, int(t_lo), int(t_hi)))

    def side_count(self, side):
        '''Number of strings of side 0 (A) or 1 (B) in the plan.'''
        return int(self._lib.zg_product_strings(self._handle, side, None))


class _SmallPlan:
    '''A space of fewer than two electrons, handled with explicit
    determinants; the plan interface for the solvers.'''

    def __init__(self, n, nelec, occslst):
        self.n, self.nelec = n, nelec
        self._occ = numpy.asarray(occslst, dtype=int).reshape(len(occslst), nelec)
        self.dim = len(self._occ)

    def occslst(self):
        return self._occ

    def info(self):
        return {}


def _chunks(orbitals, size):
    '''orbitals in ceil(len / size) consecutive pieces of nearly equal size.'''
    orbitals = list(orbitals)
    if not orbitals:
        return []
    k = -(-len(orbitals) // size)
    bounds = numpy.linspace(0, len(orbitals), k + 1).round().astype(int)
    return [orbitals[bounds[i]:bounds[i + 1]] for i in range(k)]


def _atoms(norb, sets):
    '''The orbitals grouped by their weights in the constraints (sets are
    dicts orbital -> weight), in the order of their first orbital.'''
    groups = {}
    for p in range(norb):
        key = tuple(s.get(p, 0) for s in sets)
        groups.setdefault(key, []).append(p)
    return sorted(groups.values(), key=lambda g: g[0])


def _constraint_list(gas, restrictions, norb):
    '''[(dict orbital -> weight, min, max)] from the gas and restrictions
    inputs: dict(orbitals=[...], min=..., max=..., weights=[...]), weights
    (integers, one per orbital) 1 if not given.'''
    out = []
    for item in list(gas or []) + list(restrictions or []):
        orbitals = [int(p) for p in item['orbitals']]
        weights = item.get('weights')
        weights = [1] * len(orbitals) if weights is None else [int(w) for w in weights]
        if len(weights) != len(orbitals):
            raise ValueError('weights needs one entry per orbital')
        if len(set(orbitals)) != len(orbitals):
            raise ValueError('an orbital is named twice in a constraint')
        if not orbitals or min(orbitals) < 0 or max(orbitals) >= norb:
            raise ValueError('constraint on orbitals %s out of 0..%d'
                             % (sorted(orbitals), norb - 1))
        wmap = {p: w for p, w in zip(orbitals, weights) if w != 0}
        lo = int(item.get('min', min(0, sum(w for w in weights if w < 0))))
        hi = int(item.get('max', sum(w for w in weights if w > 0)))
        out.append((wmap, lo, hi))
    return out


def _subgroups_of(sets, groups):
    '''For every constraint (dict orbital -> weight), the sub-groups in it
    and their weights; an error if a sub-group has orbitals of different
    weights.'''
    result = []
    for s in sets:
        subs, weights = [], []
        for k, g in enumerate(groups):
            w = set(s.get(p, 0) for p in g)
            if len(w) > 1:
                raise ValueError('the sub-groups must not cut a constraint: '
                                 'sub-group %s and constraint %s' % (g, s))
            w = w.pop()
            if w:
                subs.append(k)
                weights.append(w)
        result.append((subs, weights))
    return result


class GASCI(lib.StreamObject):
    '''Full, RAS and GAS CI over spinors.

    Attributes:
        nroots : int
        conv_tol : float
            Energy convergence of the Davidson iterations.
        conv_tol_residual : float or None
        max_cycle, max_space, lindep, level_shift : Davidson settings.
        gas : list of dict(orbitals=[...], min=..., max=...) or None
            GAS spaces: bounds on the number of electrons in each.  Default:
            no constraint (full CI).  Orbitals not named are free.
        restrictions : list of dict(orbitals=[...], min=..., max=...)
            More bounds, on any sets of orbitals (the format of
            SHCI.restrictions); e.g. cumulative bounds of a RAS.
        orbsym : list of int, list of such lists, or None
            Additive symmetry label of every spinor, as in SHCI.
        orbsym_mod : int, or list of int
            Labels add modulo this number; 0 for plain addition.
        target : list of int or None
            The sum of the labels of the occupied spinors, one per set.
            Default: that of the lowest nelec spinors.
        max_subgroup : int
            Largest number of spinors in a sub-group (at most 16).
        subgroups : (groups_a, groups_b) or None
            Explicit division into sub-groups, lists of lists of orbitals that
            together hold every orbital once; the constraints must be unions
            of sub-groups.
        direct_dim : int
            Spaces up to this size are diagonalized as a full matrix.
        sigma_kernel : 'blas' (default), 'block' or 'reference'
            The C kernel of H c (lib/zgasci/README.md).
    '''

    def __init__(self, mol=None):
        self.mol = mol
        if mol is None:
            self.stdout = None
            self.verbose = logger.NOTE
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
        if self.stdout is None:
            import sys
            self.stdout = sys.stdout
        self.nroots = 1
        self.conv_tol = 1e-10
        self.conv_tol_residual = None
        self.max_cycle = 100
        self.max_space = 12
        self.lindep = 1e-14
        self.level_shift = 1e-3
        self.max_memory = 4000
        self.gas = None
        self.restrictions = []
        self.orbsym = None
        self.orbsym_mod = 0
        self.target = None
        self.max_subgroup = 12
        self.subgroups = None
        self.direct_dim = 300
        self.sigma_kernel = 'blas'
        self.converged = False
        self.eci = None
        self.ci = None
        self._plan_cache = (None, None)
        # Spin is not a good quantum number for spinors; as zfci.FCISolver.
        self.spin_square = None
        self.states_spin_square = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('nroots = %d', self.nroots)
        log.info('gas = %s', self.gas)
        if self.restrictions:
            log.info('restrictions = %s', self.restrictions)
        log.info('max_subgroup = %d', self.max_subgroup)
        return self

    # -- the space ---------------------------------------------------------

    def _nelec(self, nelec):
        if isinstance(nelec, (int, numpy.integer)):
            return int(nelec)
        return int(sum(nelec))

    def _label_sets(self, norb):
        if self.orbsym is None or len(self.orbsym) == 0:
            return numpy.zeros((0, norb), dtype=int), [], None
        sets = numpy.asarray(self.orbsym, dtype=int)
        if sets.ndim == 1:
            sets = sets[None, :]
        if sets.shape[1] != norb:
            raise ValueError('orbsym needs one label per orbital')
        mods = numpy.atleast_1d(self.orbsym_mod).astype(int)
        if len(mods) == 1:
            mods = numpy.repeat(mods, len(sets))
        if len(mods) != len(sets):
            raise ValueError('orbsym_mod needs one entry per label set')
        targets = self.target
        if targets is not None:
            targets = [int(x) for x in numpy.atleast_1d(targets)]
            if len(targets) != len(sets):
                raise ValueError('target needs one entry per label set')
        return sets, [int(x) for x in mods], targets

    def layout(self, norb, nelec):
        '''The reordering and the division into sub-groups.

        Returns:
            perm : the original orbital of each spinor of the plan
            groups : the sub-groups as lists of positions in perm
            nsub_a : sub-groups on side A
            constraints : [(sub-groups, min, max)]
        '''
        cons = _constraint_list(self.gas, self.restrictions, norb)
        sets = [c[0] for c in cons]
        if self.subgroups is not None:
            groups_a, groups_b = self.subgroups
            groups = [list(map(int, g)) for g in list(groups_a) + list(groups_b)]
            nsub_a = len(groups_a)
            flat = sorted(p for g in groups for p in g)
            if flat != list(range(norb)):
                raise ValueError('subgroups must hold every orbital once')
        else:
            size = int(self.max_subgroup)
            if not 1 <= size <= 16:
                raise ValueError('max_subgroup must be 1 to 16')
            side_a, side_b = [], []
            extra_to_a = True
            for atom in _atoms(norb, sets):
                half = len(atom) // 2
                if len(atom) % 2:
                    half += extra_to_a
                    extra_to_a = not extra_to_a
                side_a.extend(_chunks(atom[:half], size))
                side_b.extend(_chunks(atom[half:], size))
            if not side_b:
                side_b.append(side_a.pop())
            groups = side_a + side_b
            nsub_a = len(side_a)
        perm = [p for g in groups for p in g]
        subs = _subgroups_of(sets, groups)
        constraints = [(sw[0], c[1], c[2], sw[1]) for sw, c in zip(subs, cons)]
        position = numpy.argsort(perm)
        groups = [[int(position[p]) for p in g] for g in groups]
        return numpy.asarray(perm), groups, nsub_a, constraints

    def plan(self, norb, nelec):
        '''The plan of the space (cached for the same settings).'''
        nelec = self._nelec(nelec)
        labels, mods, targets = self._label_sets(norb)
        key = repr((norb, nelec, self.gas, self.restrictions,
                    labels.tolist(), mods, targets, self.max_subgroup,
                    self.subgroups))
        if self._plan_cache[0] == key:
            return self._plan_cache[1]
        perm, groups, nsub_a, constraints = self.layout(norb, nelec)
        if targets is None and len(labels):
            targets = [int(x) for x in labels[:, :nelec].sum(axis=1)]
        if nelec < 2:
            plan = self._small_plan(norb, nelec, perm, constraints, groups,
                                    labels, mods, targets)
        else:
            plan = Plan(norb, nelec, [len(g) for g in groups], nsub_a,
                        constraints, labels[:, perm] if len(labels) else None,
                        mods, targets or [])
        plan.perm = perm
        self._plan_cache = (key, plan)
        return plan

    def _small_plan(self, norb, nelec, perm, constraints, groups, labels, mods,
                    targets):
        group_of = numpy.zeros(norb, dtype=int)
        for k, g in enumerate(groups):
            group_of[g] = k
        occs = []
        for occ in itertools.combinations(range(norb), nelec):
            good = True
            for subs, lo, hi, weights in constraints:
                wt = dict(zip(subs, weights))
                k = sum(wt.get(group_of[p], 0) for p in occ)
                good = good and lo <= k <= hi
            for s in range(len(labels)):
                total = sum(labels[s, perm[p]] for p in occ)
                if mods[s]:
                    good = good and (total - targets[s]) % mods[s] == 0
                else:
                    good = good and total == targets[s]
            if good:
                occs.append(occ)
        if not occs:
            raise ValueError('the CI space is empty')
        return _SmallPlan(norb, nelec, occs)

    # -- integrals ----------------------------------------------------------

    def _hamiltonian(self, h1e, eri, norb, nelec, plan):
        '''v of the plan's spinor order (see antisymmetrized).'''
        perm = plan.perm
        h1 = numpy.asarray(h1e)[numpy.ix_(perm, perm)]
        eri = numpy.asarray(eri).reshape(norb, norb, norb, norb)
        eri = eri[numpy.ix_(perm, perm, perm, perm)]
        return antisymmetrized(h1, eri, norb, nelec)

    def _small_matrix(self, h1e, eri, norb, nelec, plan):
        h1 = numpy.asarray(h1e)[numpy.ix_(plan.perm, plan.perm)]
        if nelec == 0:
            return numpy.zeros((1, 1), dtype=h1.dtype)
        occ = plan.occslst()[:, 0]
        return h1[numpy.ix_(occ, occ)]

    # -- the solver ---------------------------------------------------------

    def kernel(self, h1e, eri, norb, nelec, ci0=None, nroots=None, ecore=0,
               verbose=None, **kwargs):
        '''Returns (e, ci) for one root, (list of e, list of ci) for
        several; e includes ecore.'''
        log = logger.new_logger(self, verbose)
        if nroots is None:
            nroots = self.nroots
        nelec_total = self._nelec(nelec)
        self.norb, self.nelec = norb, nelec
        t0 = (logger.process_clock(), logger.perf_counter())
        plan = self.plan(norb, nelec)
        if plan.dim < nroots:
            raise ValueError('%d roots asked for in a space of %d determinants'
                             % (nroots, plan.dim))
        log.info('GASCI: %d determinants  %s', plan.dim, plan.info())
        if nelec_total < 2:
            hmat = self._small_matrix(h1e, eri, norb, nelec_total, plan)
            e, c = numpy.linalg.eigh(hmat)
            e, c = e[:nroots], [c[:, k] for k in range(nroots)]
            self.converged = True
        else:
            v = self._hamiltonian(h1e, eri, norb, nelec_total, plan)
            e, c = self._diagonalize(plan, v, ci0, nroots, log)
        log.timer('GASCI', *t0)
        e = numpy.asarray(e).real + numpy.real(ecore)
        if nroots == 1:
            self.eci, self.ci = e[0], c[0]
        else:
            self.eci, self.ci = e, list(c)
        return self.eci, self.ci

    def _diagonalize(self, plan, v, ci0, nroots, log):
        hdiag = plan.hdiag(v)
        if plan.dim <= self.direct_dim:
            eye = numpy.eye(plan.dim, dtype=v.dtype)
            hmat = numpy.array([plan.sigma(v, x, self.sigma_kernel) for x in eye]).T
            e, c = numpy.linalg.eigh(hmat)
            self.converged = True
            return e[:nroots], [c[:, k] for k in range(nroots)]
        x0 = self._guess(plan, hdiag, ci0, nroots, v.dtype)
        precond = lib.make_diag_precond(hdiag, self.level_shift)
        conv, e, c = lib.davidson1(
            lambda xs: [plan.sigma(v, x, self.sigma_kernel) for x in xs], x0, precond,
            tol=self.conv_tol, max_cycle=self.max_cycle,
            max_space=self.max_space, lindep=self.lindep, nroots=nroots,
            max_memory=self.max_memory, verbose=log,
            tol_residual=self.conv_tol_residual)
        self.converged = all(conv)
        if not self.converged:
            log.warn('GASCI: Davidson not converged for roots %s',
                     [k for k, x in enumerate(conv) if not x])
        return e, c

    def _guess(self, plan, hdiag, ci0, nroots, dtype):
        if ci0 is not None:
            if isinstance(ci0, numpy.ndarray) and ci0.ndim == 1:
                ci0 = [ci0]
            ci0 = [numpy.asarray(x).ravel() for x in ci0]
            if all(x.size == plan.dim for x in ci0):
                ci0 = [x.astype(numpy.result_type(dtype, x.dtype)) for x in ci0]
                if len(ci0) >= nroots:
                    return ci0[:nroots]
                start = ci0
            else:
                start = []
        else:
            start = []
        order = numpy.argsort(hdiag, kind='stable')
        rng = numpy.random.default_rng(1)
        x0 = list(start)
        for addr in order:
            if len(x0) >= nroots:
                break
            x = numpy.zeros(plan.dim, dtype=dtype)
            x[addr] = 1
            x += 1e-6 * rng.standard_normal(plan.dim)
            x0.append(x / numpy.linalg.norm(x))
        return x0

    # -- density matrices ---------------------------------------------------

    def _rdm12_plan_order(self, bra, ket, norb, nelec):
        '''dm1, dm2 of <bra| ... |ket> over the spinors of the plan.'''
        plan = self.plan(norb, nelec)
        nelec = self._nelec(nelec)
        bra, ket = numpy.asarray(bra).ravel(), numpy.asarray(ket).ravel()
        n = plan.n
        if nelec < 2:
            dtype = numpy.result_type(bra.dtype, ket.dtype)
            dm1 = numpy.zeros((n, n), dtype=dtype)
            dm2 = numpy.zeros((n,) * 4, dtype=dtype)
            if nelec == 1:
                occ = plan.occslst()[:, 0]
                dm1[numpy.ix_(occ, occ)] = numpy.outer(bra.conj(), ket)
            return dm1, dm2
        g = plan.rdm2_pairs(bra, ket)
        return unpack_rdm2(g, n, nelec)

    def _to_original(self, dm, norb, nelec):
        perm = self.plan(norb, nelec).perm
        out = numpy.empty_like(dm)
        out[numpy.ix_(*([perm] * dm.ndim))] = dm
        return out

    def make_rdm12(self, fcivec, norb, nelec, **kwargs):
        '''dm1[p,q] = <p^+ q>, dm2[p,q,r,s] = <p^+ r^+ s q>.'''
        dm1, dm2 = self._rdm12_plan_order(fcivec, fcivec, norb, nelec)
        return (self._to_original(dm1, norb, nelec),
                self._to_original(dm2, norb, nelec))

    def make_rdm1(self, fcivec, norb, nelec, **kwargs):
        '''dm1[p,q] = <p^+ q>'''
        return self.make_rdm12(fcivec, norb, nelec)[0]

    def make_rdm2(self, fcivec, norb, nelec, **kwargs):
        '''dm2[p,q,r,s] = <p^+ r^+ s q>'''
        return self.make_rdm12(fcivec, norb, nelec)[1]

    def trans_rdm12(self, cibra, ciket, norb, nelec, **kwargs):
        '''t1[p,q] = <bra| p^+ q |ket>, t2[p,q,r,s] = <bra| p^+ r^+ s q |ket>.'''
        dm1, dm2 = self._rdm12_plan_order(cibra, ciket, norb, nelec)
        return (self._to_original(dm1, norb, nelec),
                self._to_original(dm2, norb, nelec))

    def trans_rdm1(self, cibra, ciket, norb, nelec, **kwargs):
        '''t[p,q] = <bra| p^+ q |ket>.'''
        return self.trans_rdm12(cibra, ciket, norb, nelec)[0]

    def wavefunction(self, fcivec, norb, nelec):
        '''The determinants as tuples of original orbitals in increasing
        order, and the coefficients of fcivec in that convention.'''
        plan = self.plan(norb, nelec)
        occ = plan.perm[plan.occslst()]
        # Sorting the creation operators into increasing original order.
        inversions = numpy.zeros(len(occ), dtype=int)
        for i in range(occ.shape[1]):
            for j in range(i + 1, occ.shape[1]):
                inversions += occ[:, i] > occ[:, j]
        coef = numpy.asarray(fcivec).ravel() * (1 - 2 * (inversions % 2))
        return [tuple(sorted(x)) for x in occ], coef


class SpatialGASCI(GASCI):
    '''GAS CI for a non-relativistic Hamiltonian over spatial orbitals, with
    the interface and the spin-traced density matrices of
    pyscf.fci.direct_spin1 (and SpatialSHCI); a fcisolver of
    pyscf.mcscf.CASSCF.

    h1e and eri are real and over norb spatial orbitals; nelec is
    (nalpha, nbeta), or their sum with the attribute spin.  The kernels work
    over the 2 norb spin orbitals, side A the alpha and side B the beta ones;
    N_alpha is held fixed by a constraint.  gas and restrictions name spatial
    orbitals and bound the electrons of both spins together; subgroups is a
    list of lists of spatial orbitals, used for both spins.  orbsym holds the
    PySCF numbers of the irreducible representations of D2h and its
    subgroups (0 to 7), wfnsym the target (default: that of the aufbau
    determinant).  Transition density matrices follow pyscf,
    t[p,q] = <bra| q^+ p |ket>: the transpose of SpatialSHCI.trans_rdm1.
    '''

    def __init__(self, mol=None):
        super().__init__(mol)
        self.spin = None
        self.wfnsym = None
        self.spin_square = self._spin_square
        self.davidson_only = False
        self.pspace_size = 0

    def _unpack_nelec(self, nelec):
        if isinstance(nelec, (int, numpy.integer)):
            spin = self.spin if self.spin is not None else nelec % 2
            return (int(nelec) + spin) // 2, (int(nelec) - spin) // 2
        return int(nelec[0]), int(nelec[1])

    def _wfnsym_id(self):
        wfnsym = self.wfnsym
        if wfnsym is None:
            return None
        if isinstance(wfnsym, str):
            from pyscf import symm
            return symm.irrep_name2id(self.mol.groupname, wfnsym) % 10
        return int(wfnsym) % 10

    def layout(self, norb, nelec):
        nalpha, nbeta = self._unpack_nelec(nelec)
        cons = _constraint_list(self.gas, self.restrictions, norb)
        sets = [c[0] for c in cons]
        if self.subgroups is not None:
            groups = [list(map(int, g)) for g in self.subgroups]
            flat = sorted(p for g in groups for p in g)
            if flat != list(range(norb)):
                raise ValueError('subgroups must hold every orbital once')
        else:
            size = int(self.max_subgroup)
            if not 1 <= size <= 16:
                raise ValueError('max_subgroup must be 1 to 16')
            groups = []
            for atom in _atoms(norb, sets):
                groups.extend(_chunks(atom, size))
        sperm = [p for g in groups for p in g]
        subs = _subgroups_of(sets, groups)
        ng = len(groups)
        # Spin orbital i < norb: alpha of sperm[i]; norb + i: beta of it.
        constraints = [(sw[0] + [k + ng for k in sw[0]], c[1], c[2], sw[1] + sw[1])
                       for sw, c in zip(subs, cons)]
        constraints.append((list(range(ng)), nalpha, nalpha))
        position = numpy.argsort(sperm)
        groups = [[int(position[p]) for p in g] for g in groups]
        groups = groups + [[p + norb for p in g] for g in groups]
        return numpy.asarray(sperm), groups, ng, constraints

    def plan(self, norb, nelec):
        nalpha, nbeta = self._unpack_nelec(nelec)
        orbsym = None if self.orbsym is None else numpy.asarray(self.orbsym, dtype=int)
        if orbsym is not None and len(orbsym) == 0:
            orbsym = None
        wfnsym = self._wfnsym_id()
        key = repr(('spatial', norb, nalpha, nbeta, self.gas, self.restrictions,
                    None if orbsym is None else orbsym.tolist(), wfnsym,
                    self.max_subgroup, self.subgroups))
        if self._plan_cache[0] == key:
            return self._plan_cache[1]
        sperm, groups, nsub_a, constraints = self.layout(norb, (nalpha, nbeta))
        n = 2 * norb
        labels, mods, targets = numpy.zeros((0, n), dtype=int), [], []
        if orbsym is not None:
            if orbsym.size != norb or orbsym.min() < 0 or orbsym.max() > 7:
                raise ValueError('orbsym must give one irreducible representation '
                                 'of D2h or a subgroup (0 to 7) per orbital')
            if wfnsym is None:
                wfnsym = 0
                for p in list(range(nalpha)) + list(range(nbeta)):
                    wfnsym ^= int(orbsym[p])
            rows = []
            for bit in range(3):
                bits = (orbsym >> bit) & 1
                if bits.any():
                    rows.append(numpy.concatenate([bits[sperm], bits[sperm]]))
                    mods.append(2)
                    targets.append((wfnsym >> bit) & 1)
            if rows:
                labels = numpy.array(rows)
        if nalpha + nbeta < 2:
            plan = self._small_spatial(norb, nalpha, nbeta, groups, constraints,
                                       labels, targets)
        else:
            plan = Plan(n, nalpha + nbeta, [len(g) for g in groups], nsub_a,
                        constraints, labels if len(labels) else None, mods,
                        targets)
        plan.sperm = sperm
        plan.perm = numpy.concatenate([sperm, numpy.asarray(sperm) + norb])
        plan.nalpha, plan.nbeta = nalpha, nbeta
        self._plan_cache = (key, plan)
        return plan

    def _small_spatial(self, norb, nalpha, nbeta, groups, constraints, labels,
                       targets):
        group_of = numpy.zeros(2 * norb, dtype=int)
        for k, g in enumerate(groups):
            group_of[g] = k
        occs = []
        for occ in itertools.combinations(range(2 * norb), nalpha + nbeta):
            good = True
            for subs, lo, hi, weights in constraints:
                wt = dict(zip(subs, weights))
                k = sum(wt.get(group_of[p], 0) for p in occ)
                good = good and lo <= k <= hi
            for s in range(len(labels)):
                good = good and (sum(labels[s, p] for p in occ) - targets[s]) % 2 == 0
            if good:
                occs.append(occ)
        if not occs:
            raise ValueError('the CI space is empty')
        return _SmallPlan(2 * norb, nalpha + nbeta, occs)

    def _spin_orbital_integrals(self, h1e, eri, norb, plan):
        from pyscf import ao2mo
        if numpy.iscomplexobj(h1e) or numpy.iscomplexobj(eri):
            raise ValueError('integrals over spatial orbitals must be real')
        sp = plan.sperm
        h1 = numpy.asarray(h1e, dtype=float)[numpy.ix_(sp, sp)]
        eri = ao2mo.restore(1, numpy.asarray(eri, dtype=float), norb)
        eri = eri[numpy.ix_(sp, sp, sp, sp)]
        n = 2 * norb
        h = numpy.zeros((n, n))
        h[:norb, :norb] = h[norb:, norb:] = h1
        g = numpy.zeros((n, n, n, n))
        for a in (slice(0, norb), slice(norb, n)):
            for b in (slice(0, norb), slice(norb, n)):
                g[a, a, b, b] = eri
        return h, g

    def _hamiltonian(self, h1e, eri, norb, nelec, plan):
        h, g = self._spin_orbital_integrals(h1e, eri, norb, plan)
        return antisymmetrized(h, g, 2 * norb, nelec)

    def _small_matrix(self, h1e, eri, norb, nelec, plan):
        if nelec == 0:
            return numpy.zeros((1, 1))
        h, _ = self._spin_orbital_integrals(h1e, numpy.zeros((norb,) * 4), norb, plan)
        occ = plan.occslst()[:, 0]
        return h[numpy.ix_(occ, occ)]

    def _nelec(self, nelec):
        return sum(self._unpack_nelec(nelec))

    def kernel(self, h1e, eri, norb, nelec, ci0=None, nroots=None, ecore=0,
               verbose=None, **kwargs):
        if 'orbsym' in kwargs and kwargs['orbsym'] is not None:
            self.orbsym = kwargs['orbsym']
        if 'wfnsym' in kwargs and kwargs['wfnsym'] is not None:
            self.wfnsym = kwargs['wfnsym']
        return super().kernel(h1e, eri, norb, self._unpack_nelec(nelec), ci0=ci0,
                              nroots=nroots, ecore=ecore, verbose=verbose)

    def _so_rdm12(self, bra, ket, norb, nelec):
        '''Spin-orbital dm1, dm2 in the original spatial order, alpha block
        first.'''
        nelec = self._unpack_nelec(nelec)
        dm1, dm2 = self._rdm12_plan_order(bra, ket, norb, nelec)
        perm = self.plan(norb, nelec).perm
        o1 = numpy.empty_like(dm1)
        o1[numpy.ix_(perm, perm)] = dm1
        o2 = numpy.empty_like(dm2)
        o2[numpy.ix_(perm, perm, perm, perm)] = dm2
        return o1, o2

    @staticmethod
    def _spin_blocks(dm1, dm2, norb):
        a, b = slice(0, norb), slice(norb, 2 * norb)
        return ((dm1[a, a], dm1[b, b]),
                (dm2[a, a, a, a], dm2[a, a, b, b], dm2[b, b, b, b]))

    def make_rdm12s(self, fcivec, norb, nelec, **kwargs):
        '''((dm1a, dm1b), (dm2aa, dm2ab, dm2bb)) as pyscf.fci.direct_spin1:
        dm2ab[p,q,r,s] = <p_a^+ r_b^+ s_b q_a>.'''
        dm1, dm2 = self._so_rdm12(fcivec, fcivec, norb, nelec)
        return self._spin_blocks(dm1, dm2, norb)

    def make_rdm1s(self, fcivec, norb, nelec, **kwargs):
        return self.make_rdm12s(fcivec, norb, nelec)[0]

    def make_rdm12(self, fcivec, norb, nelec, **kwargs):
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = self.make_rdm12s(fcivec, norb, nelec)
        return (dm1a + dm1b,
                dm2aa + dm2ab + dm2ab.transpose(2, 3, 0, 1) + dm2bb)

    def make_rdm1(self, fcivec, norb, nelec, **kwargs):
        dm1a, dm1b = self.make_rdm1s(fcivec, norb, nelec)
        return dm1a + dm1b

    def make_rdm2(self, fcivec, norb, nelec, **kwargs):
        return self.make_rdm12(fcivec, norb, nelec)[1]

    def trans_rdm12s(self, cibra, ciket, norb, nelec, **kwargs):
        '''As pyscf.fci.direct_spin1: t1a[p,q] = <bra| q_a^+ p_a |ket> (the
        transpose of GASCI.trans_rdm1 and of SpatialSHCI.trans_rdm1),
        t2ab[p,q,r,s] = <bra| p_a^+ r_b^+ s_b q_a |ket>, t2ba likewise.'''
        dm1, dm2 = self._so_rdm12(cibra, ciket, norb, nelec)
        (t1a, t1b), (t2aa, t2ab, t2bb) = self._spin_blocks(dm1, dm2, norb)
        b, a = slice(norb, 2 * norb), slice(0, norb)
        return (t1a.T, t1b.T), (t2aa, t2ab, dm2[b, b, a, a], t2bb)

    def trans_rdm12(self, cibra, ciket, norb, nelec, **kwargs):
        (t1a, t1b), (t2aa, t2ab, t2ba, t2bb) = self.trans_rdm12s(cibra, ciket,
                                                                 norb, nelec)
        return t1a + t1b, t2aa + t2ab + t2ba + t2bb

    def trans_rdm1s(self, cibra, ciket, norb, nelec, **kwargs):
        return self.trans_rdm12s(cibra, ciket, norb, nelec)[0]

    def trans_rdm1(self, cibra, ciket, norb, nelec, **kwargs):
        '''t[p,q] = <bra| q^+ p |ket>, spin-traced, as pyscf.fci.direct_spin1.'''
        t1a, t1b = self.trans_rdm1s(cibra, ciket, norb, nelec)
        return t1a + t1b

    def _spin_square(self, fcivec, norb, nelec, *args, **kwargs):
        '''(<S^2>, 2S+1): S^2 = S_- S_+ + S_z (S_z + 1), and
        <S_- S_+> = N_beta - sum_pq <p_a^+ q_b^+ p_b q_a>.'''
        nalpha, nbeta = self._unpack_nelec(nelec)
        _, (_, dm2ab, _) = self.make_rdm12s(fcivec, norb, (nalpha, nbeta))
        sz = 0.5 * (nalpha - nbeta)
        ss = nbeta - numpy.einsum('pqqp->', dm2ab).real + sz * (sz + 1)
        s = numpy.sqrt(max(ss, 0) + 0.25) - 0.5
        return ss, 2 * s + 1

    def wavefunction(self, fcivec, norb, nelec):
        '''Determinants as tuples of spin orbitals 2P (alpha) and 2P + 1
        (beta) in increasing order, as SpatialSHCI, and the coefficients in
        that convention.'''
        nelec = self._unpack_nelec(nelec)
        plan = self.plan(norb, nelec)
        so = plan.perm[plan.occslst()]
        occ = numpy.where(so < norb, 2 * so, 2 * (so - norb) + 1)
        inversions = numpy.zeros(len(occ), dtype=int)
        for i in range(occ.shape[1]):
            for j in range(i + 1, occ.shape[1]):
                inversions += occ[:, i] > occ[:, j]
        coef = numpy.asarray(fcivec).ravel() * (1 - 2 * (inversions % 2))
        return [tuple(sorted(x)) for x in occ], coef


def _string_key(words):
    return hash(numpy.ascontiguousarray(words).tobytes())


class _ProductMixin:
    '''What the product-space solvers share: the strings (ci_strs), the
    vectors as matrices over them, the start vector from another space.'''

    def _product_words(self, norb, nside):
        if self.ci_strs is None:
            raise ValueError('a product space needs ci_strs = (strings_a, strings_b)')
        strs_a, strs_b = self.ci_strs
        return (string_words(strs_a, nside[0]), string_words(strs_b, nside[1]))

    def strings(self, norb, nelec):
        '''The A and the B strings in the order of the plan, as integers.'''
        plan = self.plan(norb, nelec)
        return words_to_ints(plan.strings(0)), words_to_ints(plan.strings(1))

    def to_matrix(self, fcivec, norb, nelec):
        '''The CI vector as the matrix C[a, b] over strings(norb, nelec) (zero
        where (a, b) is not a determinant of the space: other labels).'''
        plan = self.plan(norb, nelec)
        ia, ib = plan.det_strings()
        fcivec = numpy.asarray(fcivec).ravel()
        out = numpy.zeros((len(plan.strings(0)), len(plan.strings(1))), dtype=fcivec.dtype)
        out[ia, ib] = fcivec
        return out

    def from_matrix(self, matrix, strs, norb, nelec):
        '''A vector of the space from the matrix C[a, b] over the strings
        strs = (strings_a, strings_b) of another space (integers or occupation
        arrays as ci_strs): the elements of the determinants that both spaces
        have, zero elsewhere.  For warm starts between SQD iterations.'''
        plan = self.plan(norb, nelec)
        nside = self._nside(norb)
        maps = []
        for side in range(2):
            old = {row.tobytes(): i for i, row in
                   enumerate(string_words(strs[side], nside[side]))}
            maps.append(numpy.array([old.get(row.tobytes(), -1)
                                     for row in plan.strings(side)], dtype=numpy.int64))
        ia, ib = plan.det_strings()
        ra, rb = maps[0][ia], maps[1][ib]
        matrix = numpy.asarray(matrix)
        out = numpy.zeros(plan.dim, dtype=matrix.dtype)
        good = (ra >= 0) & (rb >= 0)
        out[good] = matrix[ra[good], rb[good]]
        return out

    def _start(self, ci0, norb, nelec):
        '''ci0 as (strings_a, strings_b, matrix) or a list of them: mapped by
        from_matrix; anything else as it is.'''
        def one(x):
            if isinstance(x, tuple) and len(x) == 3:
                return self.from_matrix(x[2], x[:2], norb, nelec)
            return x
        if isinstance(ci0, tuple) and len(ci0) == 3:
            return one(ci0)
        if isinstance(ci0, list):
            return [one(x) for x in ci0]
        return ci0


class ProductSpaceCI(_ProductMixin, SpatialGASCI):
    '''CI in a product space over spatial orbitals (SQD / QSCI, the space of
    pyscf.fci.selected_ci): every alpha string of ci_strs[0] times every beta
    string of ci_strs[1], with the interface of SpatialGASCI and
    pyscf.fci.direct_spin1 (energies, vectors, spin-traced and spin-resolved
    density matrices, spin_square, orbsym / wfnsym).

    ci_strs = (alpha strings, beta strings): integers (bit p: orbital p
    occupied, the convention of pyscf.fci.cistring) or 0/1 occupation arrays
    (n, norb); each string must have nalpha (nbeta) electrons, repeats are
    an error.  CI vectors are in the order of the plan; without orbsym it is
    the matrix C[a, b] over the sorted strings, as in selected_ci
    (to_matrix / from_matrix convert in general).  ci0 may be such vectors or
    (strings_a, strings_b, matrix) of another space (a warm start).
    spin_flip = 1 (-1): with the same alpha and beta strings and nalpha =
    nbeta, only states with C[b, a] = C[a, b] (= -C[a, b]), i.e. even
    (odd) S, and a sigma without the beta-beta half.  Only the BLAS kernel
    (lib/zgasci/PRODUCT_SPACE.md); gas, restrictions, subgroups do not
    apply.
    ss_penalty = shift (with ss_value, default S_z (S_z + 1)): the spin
    penalty of pyscf.fci.addons.fix_spin_ (as SQD uses it): H + shift (S^2 -
    ss) if ss_value is the lowest possible value (within 0.1), else H + shift
    (S^2 - ss)^2.  S^2 is applied by the same kernel as H, from its
    coefficients over the strings (s2_antisymmetrized); spin_flip still
    applies.  The energies returned are <H> of the vectors, the eigenvalues
    of the penalized operator are in self.e_penalized.
    '''

    def __init__(self, mol=None, ci_strs=None):
        super().__init__(mol)
        self.ci_strs = ci_strs
        self.spin_flip = None
        self.ss_penalty = None
        self.ss_value = None
        self.e_penalized = None

    def _nside(self, norb):
        return (norb, norb)

    def contract_ss(self, fcivec, norb, nelec):
        '''S^2 |c> in the space (the vector in the order of the plan).'''
        plan = self.plan(norb, nelec)
        vs = s2_antisymmetrized(norb, plan.nalpha, plan.nbeta)
        return plan.sigma(vs, numpy.asarray(fcivec).ravel())

    def _diagonalize(self, plan, v, ci0, nroots, log):
        '''With spin_flip = +1 or -1 and the same alpha and beta strings
        (nalpha = nbeta): only vectors with C[b, a] = spin_flip C[a, b] (even
        or odd S), and sigma without its beta-beta half.  With ss_penalty:
        the penalized operator (see the class), then <H> of the vectors.'''
        if self.ss_penalty is None:
            self.e_penalized = None
            return self._diagonalize_v(plan, v, ci0, nroots, log)
        norb = len(plan.sperm)
        nelec = plan.nalpha + plan.nbeta
        sz = 0.5 * abs(plan.nalpha - plan.nbeta)
        ss = sz * (sz + 1) if self.ss_value is None else float(self.ss_value)
        shift = float(self.ss_penalty)
        vs = s2_antisymmetrized(norb, plan.nalpha, plan.nbeta)
        one = antisymmetrized(numpy.eye(2 * norb) / nelec, numpy.zeros((2 * norb,) * 4),
                              2 * norb, nelec)
        if ss < sz * (sz + 1) + .1:
            e, c = self._diagonalize_v(plan, v + shift * (vs - ss * one), ci0, nroots, log)
        else:
            def extra(x, sig):
                t = sig(vs, x) - ss * x
                return shift * (sig(vs, t) - ss * t)
            hs = plan.hdiag(vs) - ss
            e, c = self._diagonalize_v(plan, v, ci0, nroots, log, extra=extra,
                                       hdiag_extra=shift * hs * hs)
        self.e_penalized = numpy.asarray(e)
        e = [numpy.vdot(x, plan.sigma(v, x)).real / numpy.vdot(x, x).real for x in c]
        return e, c

    def _diagonalize_v(self, plan, v, ci0, nroots, log, extra=None, hdiag_extra=None):
        '''The Davidson of _diagonalize for the operator v (+ extra(x, sigma),
        sigma(v, x) the sigma of the space, with hdiag_extra added to the
        diagonal of the preconditioner).'''
        parity = self.spin_flip
        if parity is None and extra is None:
            return SpatialGASCI._diagonalize(self, plan, v, ci0, nroots, log)
        if parity is None:
            sig = lambda vv, x: plan.sigma(vv, x, self.sigma_kernel)
            hdiag = plan.hdiag(v) + hdiag_extra
            x0 = self._guess(plan, hdiag, ci0, nroots, v.dtype)
            conv, e, c = lib.davidson1(
                lambda xs: [sig(v, x) + extra(x, sig) for x in xs], x0,
                lib.make_diag_precond(hdiag, self.level_shift), tol=self.conv_tol,
                max_cycle=self.max_cycle, max_space=self.max_space, lindep=self.lindep,
                nroots=nroots, max_memory=self.max_memory, verbose=log,
                tol_residual=self.conv_tol_residual)
            self.converged = all(conv)
            if not self.converged:
                log.warn('ProductSpaceCI: Davidson not converged for roots %s',
                         [k for k, x in enumerate(conv) if not x])
            return e, c
        if parity not in (1, -1):
            raise ValueError('spin_flip must be None, 1 or -1')
        if not (plan.mirror and plan.nalpha == plan.nbeta):
            raise ValueError('spin_flip needs the same alpha and beta strings')
        flip = plan.flip_index()

        def sym(x):
            y = plan.flip_vector(x)
            y *= parity
            y += x
            y *= 0.5
            return y
        def sig(vv, x):
            out = plan.sigma_flip(vv, x, parity)
            return sym(plan.sigma(vv, x)) if out is None else out
        hdiag = plan.hdiag(v)
        if hdiag_extra is not None:
            hdiag = hdiag + hdiag_extra
        # Basis of the subspace: (e_k + parity e_flip(k)) / sqrt(2), k <= flip(k).
        keep = numpy.arange(plan.dim) <= flip
        if parity < 0:
            keep &= numpy.arange(plan.dim) != flip
        nsub = int(keep.sum())
        if nsub < nroots:
            raise ValueError('%d roots asked for in a subspace of %d' % (nroots, nsub))
        if nsub <= self.direct_dim:
            basis = numpy.zeros((nsub, plan.dim))
            for i, k in enumerate(numpy.nonzero(keep)[0]):
                basis[i, k] = 1
                basis[i, flip[k]] += parity
                basis[i] /= numpy.linalg.norm(basis[i])
            hmat = numpy.array([basis @ (plan.sigma(v, x) if extra is None else
                                         sig(v, x) + extra(x, sig)) for x in basis])
            e, c = numpy.linalg.eigh(hmat)
            self.converged = True
            return e[:nroots], [basis.T @ c[:, k] for k in range(nroots)]

        def hop(x):
            x = sym(x)
            out = sig(v, x)
            if extra is not None:
                out += extra(x, sig)
            return out
        if ci0 is not None:
            if isinstance(ci0, numpy.ndarray) and ci0.ndim == 1:
                ci0 = [ci0]
            ci0 = [sym(numpy.asarray(x).ravel()) for x in ci0]
            ci0 = [x / numpy.linalg.norm(x) for x in ci0 if numpy.linalg.norm(x) > 1e-8]
        x0 = list(ci0 or [])[:nroots]
        rng = numpy.random.default_rng(1)
        for k in numpy.argsort(hdiag, kind='stable'):
            if len(x0) >= nroots:
                break
            if not keep[k]:
                continue
            x = 1e-6 * rng.standard_normal(plan.dim)
            x[k] += 1
            x[flip[k]] += parity
            x = sym(x)
            x0.append(x / numpy.linalg.norm(x))
        # The new directions symmetrized as well: otherwise the
        # antisymmetric rounding noise, untouched by hop, grows by orders of
        # magnitude per iteration once the residual is small.
        base = lib.make_diag_precond(hdiag, self.level_shift)

        def precond(dx, e, *args):
            return sym(base(dx, e, *args))
        conv, e, c = lib.davidson1(
            lambda xs: [hop(x) for x in xs], x0, precond, tol=self.conv_tol,
            max_cycle=self.max_cycle, max_space=self.max_space, lindep=self.lindep,
            nroots=nroots, max_memory=self.max_memory, verbose=log,
            tol_residual=self.conv_tol_residual)
        self.converged = all(conv)
        if not self.converged:
            log.warn('ProductSpaceCI: Davidson not converged for roots %s',
                     [k for k, x in enumerate(conv) if not x])
        return e, c

    def plan(self, norb, nelec):
        nalpha, nbeta = self._unpack_nelec(nelec)
        words = self._product_words(norb, (norb, norb))
        for side, (w, k) in enumerate(zip(words, (nalpha, nbeta))):
            counts = numpy.unpackbits(numpy.ascontiguousarray(w).view(numpy.uint8),
                                      axis=1).sum(axis=1)
            if len(w) and (counts != k).any():
                raise ValueError('ci_strs[%d]: %d strings without %d electrons'
                                 % (side, int((counts != k).sum()), k))
        orbsym = None if self.orbsym is None else numpy.asarray(self.orbsym, dtype=int)
        if orbsym is not None and len(orbsym) == 0:
            orbsym = None
        wfnsym = self._wfnsym_id()
        key = repr(('product', norb, nalpha, nbeta, _string_key(words[0]),
                    _string_key(words[1]), None if orbsym is None else orbsym.tolist(),
                    wfnsym))
        if self._plan_cache[0] == key:
            return self._plan_cache[1]
        if nalpha + nbeta < 2:
            raise ValueError('a product space needs at least two electrons')
        n = 2 * norb
        labels, mods, targets = None, [], []
        if orbsym is not None:
            if orbsym.size != norb or orbsym.min() < 0 or orbsym.max() > 7:
                raise ValueError('orbsym must give one irreducible representation '
                                 'of D2h or a subgroup (0 to 7) per orbital')
            if wfnsym is None:
                wfnsym = 0
                for p in list(range(nalpha)) + list(range(nbeta)):
                    wfnsym ^= int(orbsym[p])
            rows = []
            for bit in range(3):
                bits = (orbsym >> bit) & 1
                if bits.any():
                    rows.append(numpy.concatenate([bits, bits]))
                    mods.append(2)
                    targets.append((wfnsym >> bit) & 1)
            if rows:
                labels = numpy.array(rows)
        plan = ProductPlan(n, nalpha + nbeta, norb, words[0], words[1], labels, mods,
                           targets)
        plan.sperm = numpy.arange(norb)
        plan.perm = numpy.arange(n)
        plan.nalpha, plan.nbeta = nalpha, nbeta
        self._plan_cache = (key, plan)
        return plan

    def kernel(self, h1e, eri, norb, nelec, ci0=None, nroots=None, ecore=0,
               verbose=None, ci_strs=None, **kwargs):
        if ci_strs is not None:
            self.ci_strs = ci_strs
        if 'orbsym' in kwargs and kwargs['orbsym'] is not None:
            self.orbsym = kwargs['orbsym']
        if 'wfnsym' in kwargs and kwargs['wfnsym'] is not None:
            self.wfnsym = kwargs['wfnsym']
        ci0 = self._start(ci0, norb, nelec)
        return SpatialGASCI.kernel(self, h1e, eri, norb, nelec, ci0=ci0, nroots=nroots,
                                   ecore=ecore, verbose=verbose)


class ProductSpinorCI(_ProductMixin, GASCI):
    '''CI in a product space over spinors: every A string of ci_strs[0]
    (over the spinors 0 .. nside_a - 1) times every B string of ci_strs[1]
    (over the spinors nside_a .. norb - 1, bit j: spinor nside_a + j), with
    nelec electrons together and, if orbsym is given, the labels adding up to
    target (as GASCI; 2 Omega, parity, ...).  The strings of a side may have
    different electron counts.  nside_a defaults to norb // 2.  The interface
    of GASCI (density matrices in its conventions); vectors as in
    ProductSpaceCI.
    '''

    def __init__(self, mol=None, ci_strs=None, nside_a=None):
        super().__init__(mol)
        self.ci_strs = ci_strs
        self.nside_a = nside_a

    def _nside(self, norb):
        a = norb // 2 if self.nside_a is None else int(self.nside_a)
        return (a, norb - a)

    def plan(self, norb, nelec):
        nelec = self._nelec(nelec)
        nside = self._nside(norb)
        words = self._product_words(norb, nside)
        labels, mods, targets = self._label_sets(norb)
        key = repr(('product', norb, nelec, nside, _string_key(words[0]),
                    _string_key(words[1]), labels.tolist(), mods, targets))
        if self._plan_cache[0] == key:
            return self._plan_cache[1]
        if nelec < 2:
            raise ValueError('a product space needs at least two electrons')
        if targets is None and len(labels):
            targets = [int(x) for x in labels[:, :nelec].sum(axis=1)]
        plan = ProductPlan(norb, nelec, nside[0], words[0], words[1],
                           labels if len(labels) else None, mods, targets or [])
        plan.perm = numpy.arange(norb)
        self._plan_cache = (key, plan)
        return plan

    def kernel(self, h1e, eri, norb, nelec, ci0=None, nroots=None, ecore=0,
               verbose=None, ci_strs=None, **kwargs):
        if ci_strs is not None:
            self.ci_strs = ci_strs
        ci0 = self._start(ci0, norb, nelec)
        return GASCI.kernel(self, h1e, eri, norb, nelec, ci0=ci0, nroots=nroots,
                            ecore=ecore, verbose=verbose)

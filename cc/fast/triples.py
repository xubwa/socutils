'''The (T) correction of the spinor CCSD in batches of virtual pairs.

The equations are those of PySCF's gccsd_t (and of zccsdt.zccsd.gccsd_t):
with <bc||ei> and <ma||jk> the integrals and D the orbital-energy
denominator,

    x[ijk,abc] = sum_e t2[j,k,a,e] <bc||ei> - sum_m t2[i,m,b,c] <ma||jk>
    w = P(a/bc) P(i/jk) x,      v = w + P(a/bc) P(i/jk) (t1[i,a] <bc||jk> + f[a,i] t2[j,k,b,c])
    E(T) = sum_{i>j>k, a>b>c} conj(v) w / D.

x is antisymmetric in (j,k) and in (b,c), so everything is kept on packed
pairs (numpy.tril_indices order):

    x_{a,(bc)}[i, (jk)] = sum_e <bc||ei> t2[j,k,a,e] - sum_m t2[i,m,b,c] <ma||jk>

is, for fixed a and pair (b>c), an o x o(o-1)/2 matrix, and for a > b > c

    Z[i, (jk)] = x_{a,(bc)} - x_{b,(ac)} + x_{c,(ab)},
    w[i>j>k]   = Z[i, (jk)] - Z[j, (ik)] + Z[k, (ij)].

The triples are done in batches - a range of a with all its b if that
fits, else one a and a range of b - (all c < b).  With the two arrays

    A[(b>c), i, (e | m)] = (<bc||ei> | t2[i,m,b,c])
    B[a, (j>k), (e | m)] = (t2[j,k,a,e] | -<ma||jk>)

(<bc||ei> is the stored <ie||bc>, transposed once) the particle and the
hole term of x are one product over the joint index (e | m), and the x of a
batch are two or three GEMMs on contiguous blocks of A and B: x_{a,(bc)} for
the pairs of the batch, and x_{y,(ax)} for the rectangles of (x, y) it needs
(o^3 v^4 / 4 multiply-adds for the particle term in all, each x made once
and written once).  An OpenMP kernel (csrc/zccsd_t.c) assembles Z for one
(b, c) per thread in its cache and adds up the energy.

With ``three_m`` A and B are kept as their real parts (Re, Im, Re + Im) and
every complex product is three real GEMMs, A_r B_r, A_i B_i and
(A_r + A_i)(B_r + B_i), instead of four; the three real results of a batch
go to the kernel as they are.  One and a half times the memory for a
quarter fewer flops: the default where the real parts of both operands and
a whole batch fit in memory.

Kramers pairs.  For spinors in time-reversal pairs, ordered [unbarred,
barred] in the virtual space (zccsdt.zccsd.kramers), the triples (a, b, c)
and (Ka, Kb, Kc) contribute complex-conjugate energies, and K turns a
triple with n barred virtuals into one with 3 - n: the triples with at most
one barred virtual - b in the unbarred half, whatever a - are one of each
pair.  ``kramers`` restricts the batches to them and doubles the sum: half
the GEMMs.  (For a range of barred a the pairs (b, c) are the same, those
below the barred half: one product for the x_{a,(bc)} of the range.)

Memory: A and B are kept in memory, the smaller first, as long as they fit
next to the smallest batch; else they are mapped arrays of a scratch file,
written and read in contiguous blocks.  The batches are sized from what
``max_memory`` leaves (at most BATCH_BYTES), and their x are written into
the same arrays, allocated once for the largest batch (new arrays for every
batch are mapped and zeroed by the system page by page: a fifth of the time
of the GEMMs).  MemoryError below three packed T2 and the batch of one b.

The (T) terms of the Lambda equations and of the one-particle density
(``terms=True``; the equations are in zccsd.lam_t) are sums over the same
triples, each in three roles - one of its virtual indices s, the other two as
a pair p > q, the sign of the permutation (s, p, q) of (a, b, c).  Two of
them cost as much as the x: with w and u the connected triples and those
plus the disconnected ones, over the denominator, and M[i, (jk)] =
conj(w + u)[ijk],

    lt[(se), (ij)] += sign sum_{(pq), k}  A[(pq), k, e]      M[k, (ij)]     (- (s <-> e))
    lt[(pq), (ij)] += sign sum_{s, (mn)}  M[i, (mn)] B[s, (mn), v + j]     (- (i <-> j))

(A[(pq), k, e] = -conj(<ke||pq>), B[s, (mn), v + j] = -conj(<mn||js>): the
integrals in A and B).  Each of the three x of a triple - x_{a,(bc)},
x_{b,(ac)}, x_{c,(ab)} - is the x of that triple alone, and the three are its
three roles: the kernel writes sign M over them once it has formed w, and
the two sums become the products of the batch transposed,

    x = A B^T      ->      Q = A[:, :v]^T M,     Y = M B[:, v:],

for every product that made x: Q[e, (s, (ij))] summed over the rows (pair,
k) of the product, Y[(pair, i), j] over its columns (s, (mn)).  They leave
GEMM in an array of REVERSE_BYTES at most, a block of s or of pairs at a
time, and are antisymmetrized into lt[(a>b), (i>j)], the transpose of the
packed layout (rows that are written as a whole).  The other terms - on
<bc||jk>, on t2, on the Fock matrix - are sums in the kernel.  With Kramers
pairs the triples that are done give half of every sum, the time-reversed
image the other half.

The densities of (T) for the orbital gradient (``terms='gradient'``; the
equations are in zccsd.lam_t.densities, their use in zccsd.orbgrad_t) are
the derivatives of the same sums with respect to the integrals instead of
t2, and A and B hold both: the two products of the batch transposed then
run over all of A and of B,

    Q = A^T M:   the rows of A[:, v:] = t2 give minus the density of <ma||jk>,
    Y = M B:     the columns of B[:, :v] = t2 give the density of <bc||ek>,

for v + o instead of v and o multiply-adds per element of M - the
arithmetic of the forward product in both passes, so that the terms cost
twice the products of the energy.  The density of <bc||jk> is the sum of
the f term of lambda2 with t1[k, s] in the place of f[s, k].  The density
of <bc||ek>, 8 o v^3 bytes, is added up by rows of pairs like A (in memory
after the operands if it fits, else in the scratch file) and sorted into
the layout of the stored <ke||bc> at the end.

Timers (zccsdt.timing): 't.setup', 't.gemm', 't.kernel'; 't.lam.gemm' and
't.lam.add' for the two products of the Lambda terms, 't.den.sort'.
'''

import ctypes

import numpy as np
from pyscf.lib import logger

from socutils.cc.fast import blas, memory, scratch
from socutils.cc.fast.timing import timer
from socutils.cc.fast import kernels as K

# The x of a batch, at most.  Benzene/cc-pVDZ with Kramers pairs (job 31556395):
# 252.6 / 233.5 / 228.2 / 228.7 / 228.8 s for 0.5 / 1 / 2 / 4 / 8 GB with ZGEMM,
# 234.1 / 223.5 / 223.9 s for 2 / 4 / 8 GB with three real GEMMs (which hold two
# thirds of the elements in the same bytes)
BATCH_BYTES = 4 << 30
# The products of the triples with the integrals (the terms of Lambda) of a
# block of virtuals or of pairs, at most
REVERSE_BYTES = 64 << 20
# Of what max_memory leaves beyond the least, this much is not given to the batches
MARGIN_BYTES = 32 << 20

_P, _I = ctypes.c_void_p, ctypes.c_int64
if K._lib is not None and hasattr(K._lib, 'zc_t_batch'):
    K._lib.zc_t_batch.argtypes = [_P, _I, _I, _P, _I, _P, _I, ctypes.c_int, _I, _I, _I, _I, _I,
                                  _P, _P, _P, _P, _P, _P, _P]
    K._lib.zc_t_batch.restype = None
    for _f, _a in (('zc_t_ovvv', [_P, _I, _I, _I, _I, _I, _I, _I, ctypes.c_int, _P, _P, _P]),
                   ('zc_t_t2u', [_P, _I, _I, _I, _I, _I, _I, ctypes.c_int, _P, _P, _P]),
                   ('zc_t_t2x', [_P, _I, _I, _I, _I, _I, _I, ctypes.c_int, _P, _P, _P])):
        getattr(K._lib, _f).argtypes, getattr(K._lib, _f).restype = _a, None
    _HAVE_C = True
else:
    _HAVE_C = False
_HAVE_LAM = _HAVE_C and hasattr(K._lib, 'zc_t_den_hole')     # not in a library built before it
if _HAVE_LAM:
    for _f, _a in (('zc_t_batch_lam', K._lib.zc_t_batch.argtypes + [_P] * 6),
                   ('zc_t_lam_particle', [_P, _P, _P, ctypes.c_int, _I, _I, _I, _I, _I, _P]),
                   ('zc_t_lam_hole', [_P, _P, _P, ctypes.c_int, _I, _I, _I, _I, _I, _P]),
                   ('zc_t_den_rows', [_P, _P, _P, ctypes.c_int, _I, _I, _I, _P]),
                   ('zc_t_den_hole', [_P, _P, _P, ctypes.c_int, _I, _I, _I, _I, _I, _I, _P]),
                   ('zc_t_den_sort', [_P, _I, _I, _I, _I, _I, _P])):
        getattr(K._lib, _f).argtypes, getattr(K._lib, _f).restype = _a, None


def _complex(x):
    '''A complex array from its three real products (3M), or as it is.'''
    if isinstance(x, (tuple, list)):
        return (x[0] - x[1]) + 1j * (x[2] - x[0] - x[1])
    return x


class _Sums:
    '''The (T) terms of the Lambda equations and of the density as they are
    added up over the triples: l1[i,a], dvo[a,i], the diagonals goo[i] and
    gvv[a], and lt[(a>b), (i>j)], the terms of lambda2 transposed.  With
    ``densities``: g2[m, a, (j>k)] and g3[(b>c), (j>k)], the densities of
    <ma||jk> and <bc||jk>; g1[(b>c), k, e], that of <bc||ek>, is given by
    kernel(), which decides where it lives.'''

    def __init__(self, o, v, densities=False):
        self.l1 = np.zeros((o, v), dtype=np.complex128)
        self.dvo = np.zeros((v, o), dtype=np.complex128)
        self.goo = np.zeros(o, dtype=np.complex128)
        self.gvv = np.zeros(v, dtype=np.complex128)
        self.lt = np.zeros((K.npair(v), K.npair(o)), dtype=np.complex128)
        self.g1 = self.g2 = self.g3 = None
        if densities:
            self.g2 = np.zeros((o, v, K.npair(o)), dtype=np.complex128)
            self.g3 = np.zeros((K.npair(v), K.npair(o)), dtype=np.complex128)


class Densities:
    '''The densities of (T) for the orbital gradient (zccsd.orbgrad_t), the
    coefficients of the integrals in the first-order change of Re E(T)
    (lam_t.densities has them as full arrays; here with the weight of the
    packed pairs):

      g1[k, e, (b>c)]   of <bc||ek>, in the layout of the stored <ke||bc>;
      g2[m, a, (j>k)]   of <ma||jk>;
      g3[(j>k), (b>c)]  of <bc||jk>, packed like t2,

    in memory, or all three mapped arrays of a scratch file if the first
    does not fit.  close() releases the scratch file.'''

    def __init__(self, g1, g2, g3, scr=None):
        self.g1, self.g2, self.g3, self._scr = g1, g2, g3, scr

    def close(self):
        self.g1 = None
        if self._scr is not None:
            self.g2 = self.g3 = None
            self._scr.close()
            self._scr = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _t_batch(X1, c1, G1, g1, G2, a, b0, b1, t1, fvo, VpT, T2pT, eo, ev, terms=None, sums=None):
    '''The (T) energy of the triples (a, b, c), b0 <= b < b1, c < b, from
    matrices with rows (pair or virtual, i) and columns (virtual, (jk)):

    X1[(p - p0, i), c1 + (jk)] = x_{a,(bc)}, p the pair (b, c), p0 the pair (b0, 0);
    G1[g1 + (x, i), (y - b0, (jk))] = x_{y,(ax)} for x < b1 and y >= b0;
    G2[(x - b0, i), (y, (jk))] = x_{y,(ax)} for b0 <= x < b1 and y < b0.

    The three are complex arrays or tuples of the three real products of
    the 3M scheme.  Returns a complex number whose imaginary part should
    vanish.  ``terms``: an object whose add(a, b, c, w, u) gets the
    connected triples w and w plus the disconnected ones u of every virtual
    triple, divided by the denominator and packed over i > j > k (the (T)
    terms of the Lambda equations one triple at a time: zccsd.lam_t.Terms,
    the reference).  ``sums``: the _Sums that these terms are added to; then
    the three x of every triple are overwritten by M[i, (jk)] =
    conj(w + u)[ijk] with the sign of their role - x_{a,(bc)} by M,
    x_{b,(ac)} by -M, x_{c,(ab)} by M -, and the x_{y,(ay)} of G1 by zero:
    the operands of the two products that kernel() does with them.'''
    o, v = t1.shape
    npo = K.npair(o)
    m3 = isinstance(X1, (tuple, list))
    first = lambda x: x[0] if m3 else x                         # noqa: E731
    s1, n1 = first(X1).shape[1], first(G1).shape[1]
    n2 = first(G2).shape[1] if b0 else 0
    if _HAVE_C and K.USE_C and terms is None and (sums is None or _HAVE_LAM):
        out = np.zeros(2)
        c = K._c
        dt = np.float64 if m3 else np.complex128

        def ptrs(x, row=0):
            x = (x if m3 else (x,)) if x is not None else ()
            p = [c(y, dt) + row * y.strides[0] for y in x] + [None] * 3
            return (ctypes.c_void_p * 3)(*p[:3])
        args = (ptrs(X1), s1, c1, ptrs(G1, g1), n1, ptrs(G2 if b0 else None), n2,
                int(m3), a, b0, b1, o, v, c(t1), c(fvo), c(VpT), c(T2pT),
                c(eo, np.float64), c(ev, np.float64), c(out, np.float64))
        if sums is None:
            K._lib.zc_t_batch(*args)
        else:
            K._lib.zc_t_batch_lam(*args, c(sums.l1), c(sums.dvo), c(sums.goo), c(sums.gvv),
                                  c(sums.lt), None if sums.g3 is None else c(sums.g3))
        return complex(out[0], out[1])
    p0, pa = b0 * (b0 - 1) // 2, a * (a - 1) // 2
    if sums is not None:        # the x as they are, to be written
        raw = lambda x: x if m3 else (x,)                           # noqa: E731
        X1w = [x.reshape(-1, o, s1)[:, :, c1:c1 + npo] for x in raw(X1)]
        G1w = [x[g1:g1 + b1 * o].reshape(b1, o, n1 // npo, npo) for x in raw(G1)]
        G2w = [x.reshape(b1 - b0, o, b0, npo) for x in raw(G2)] if b0 else None
    X1 = _complex(X1).reshape(-1, o, s1)[:, :, c1:c1 + npo]
    G1 = _complex(G1)[g1:g1 + b1 * o].reshape(b1, o, n1 // npo, npo)
    G2 = _complex(G2).reshape(b1 - b0, o, b0, npo) if b0 else None
    i, j, k = np.array([(i, j, k) for i in range(o) for j in range(i) for k in range(j)]).T
    pjk, pik, pij = j * (j - 1) // 2 + k, i * (i - 1) // 2 + k, i * (i - 1) // 2 + j
    e = 0j
    for b in range(max(b0, 1), b1):
        for c in range(b):
            pbc = b * (b - 1) // 2 + c
            Z = X1[pbc - p0] - G1[c, :, b - b0]
            Z = Z + (G1[b, :, c - b0] if c >= b0 else G2[b - b0, :, c])
            V = (Z + t1[:, a, None] * VpT[pbc] + fvo[a][:, None] * T2pT[pbc]
                 - t1[:, b, None] * VpT[pa + c] - fvo[b][:, None] * T2pT[pa + c]
                 + t1[:, c, None] * VpT[pa + b] + fvo[c][:, None] * T2pT[pa + b])
            w = Z[i, pjk] - Z[j, pik] + Z[k, pij]
            u = V[i, pjk] - V[j, pik] + V[k, pij]
            d = eo[i] + eo[j] + eo[k] - ev[a] - ev[b] - ev[c]
            e += np.sum(u.conj() * w / d)
            if terms is not None:
                terms.add(a, b, c, w / d, u / d)
            if sums is not None:
                wd, ud = w / d, u / d
                e3 = ud.conj() * wd
                for x in (i, j, k):
                    np.add.at(sums.goo, x, e3)
                sums.gvv[[a, b, c]] += e3.sum().conj()
                # conj(w) and M = conj(w + u) as [i, (jk)], all i
                Wc, M = (np.zeros((o, npo), dtype=np.complex128) for _ in range(2))
                for full, val in ((Wc, wd.conj()), (M, (wd + ud).conj())):
                    full[i, pjk], full[j, pik], full[k, pij] = val, -val, val
                for s, row, sign in ((a, pbc, 1.0), (b, pa + c, -1.0), (c, pa + b, 1.0)):
                    sums.l1[:, s] += sign * (Wc @ VpT[row])
                    sums.dvo[s] += sign * (Wc @ T2pT[row]).conj()
                    sums.lt[row] += sign * (fvo[s] @ Wc)
                    if sums.g3 is not None:
                        sums.g3[row] += sign * (t1[:, s] @ Wc)
                third = ([x[b, :, c - b0] for x in G1w] if c >= b0
                         else [x[b - b0, :, c] for x in G2w])
                for out, val in (([x[pbc - p0] for x in X1w], M),
                                 ([x[c, :, b - b0] for x in G1w], -M), (third, M)):
                    _store(val, out)
    if sums is not None:
        for y in range(b0, b1):
            for x in G1w:
                x[y, :, y - b0] = 0.0
    return e


def _a_range(a0, v, o, nbytes, xbytes=16):
    '''The largest a1 such that the arrays of the batch a0 <= a < a1 with all
    b < a (x_{a,(bc)} for the pairs below a1 - 1, x_{y,(ax)} for x < a and
    y < a1 - 1) take ``nbytes``; a0 if not even one a fits.'''
    unit = xbytes * o * K.npair(o)

    def size(a1):
        return unit * (K.npair(a1 - 1) * (a1 - a0) + (K.npair(a1) - K.npair(a0)) * (a1 - 1))
    a1 = a0
    while a1 < v and size(a1 + 1) <= nbytes:
        a1 += 1
    return a1


def _batches(a, o, nbytes, xbytes=16):
    '''Ranges (b0, b1) of b < a whose arrays (x_{a,(bc)} for the pairs with b
    in the range, and the two rectangles of x_{y,(ax)}) take ``nbytes``.'''
    unit = xbytes * o * K.npair(o)
    out, b0 = [], 0
    while b0 < a:
        b1 = b0 + 1
        while b1 < a and unit * (K.npair(b1 + 1) - K.npair(b0) + (b1 + 1) * (b1 + 1 - b0)
                                 + (b1 + 1 - b0) * b0) <= nbytes:
            b1 += 1
        out.append((b0, b1))
        b0 = b1
    return out


def _put(name, src, args, out, m3):
    '''One block of an operand through its kernel, or with numpy.'''
    if _HAVE_C and K.USE_C:
        dt = np.float64 if m3 else np.complex128
        ptrs = [K._c(x, dt) for x in out] + [None, None]
        getattr(K._lib, name)(src, *args, int(m3), *ptrs[:3])
        return True
    return False


def _store(x, out):
    '''The complex block x into out: one complex array, or the three real
    parts of the 3M scheme.'''
    if len(out) == 1:
        out[0][...] = x
    else:
        out[0][...], out[1][...] = x.real, x.imag
        np.add(out[0], out[1], out=out[2])


def memory_need(nocc, nvir, three_m=True, terms=False):
    '''(bytes of the arrays that are always in memory - two packed T2 and a
    third while t2 is being packed -, bytes of the batch of one virtual
    orbital): the least (T) can run in is their sum.  ``terms``: with the
    terms of the Lambda equations - their array takes the place of the third
    packed T2, and the products of the triples with the integrals need an
    array for one virtual orbital or one pair; 'gradient': with the
    densities as well - a fourth packed array, the density of <ma||jk>, and
    the products as wide as the operands (the density of <bc||ek> is in
    memory only if there is room for it).'''
    npo, npv, xbytes = K.npair(nocc), K.npair(nvir), 24 if three_m else 16
    fixed, floor = 48 * npo * npv, xbytes * nocc * npo * 3 * nvir
    if terms == 'gradient':
        n = nocc + nvir
        fixed += 16 * npo * (npv + nocc * nvir)
        floor += xbytes * max(n * npo, nocc * n)
    elif terms:
        floor += xbytes * max(nvir * npo, nocc * nocc)
    return fixed, floor


def kernel(mycc, eris, t1=None, t2=None, verbose=None, max_memory=None, three_m=None,
           kramers=None, terms=None):
    '''E(T) for the amplitudes t1 and t2 (t2[i,j,a,b] or packed
    [(i>j), (a>b)]) and the integrals of DirectZCCSD.  ``three_m``: three
    real GEMMs for each complex product (default: if the real parts of both
    operands and a whole batch fit in memory).  ``max_memory`` in MB (default: that of
    ``mycc``).  ``kramers``: the spinors are Kramers pairs in the
    order [unbarred, barred] and the amplitudes have their symmetry: only
    the triples with at most one barred virtual are done (default: if
    ``eris.kramers`` says so and the amplitudes pass kramers.symmetric).

    ``terms=True``: with the (T) terms of the Lambda equations and of the
    one-particle density, (E(T), (l1, l2p, goo, gvv, dvo)): the source
    terms of lambda1[i,a] and of lambda2 packed [(i>j), (a>b)] before the
    division by their denominators, and the terms of the density of
    zccsd.lam_t.gamma1 (about twice the time of E(T), a packed T2 more in
    memory at the end).  ``terms='gradient'``: these and the densities of
    (T) for the orbital gradient, (E(T), (l1, l2p, goo, gvv, dvo), Densities)
    - the two products of the triples of a batch then run over all of A and
    of B, the t2 in them as well as the integrals.  ``terms`` may also be an
    object as _t_batch takes it (the reference, one triple at a time);
    terms.kramers is set to what is used here.'''
    log = logger.new_logger(mycc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    t1 = mycc.t1 if t1 is None else t1
    t2 = mycc.t2 if t2 is None else t2
    t1 = np.ascontiguousarray(t1, dtype=np.complex128)
    o, v = t1.shape
    npo, npv = K.npair(o), K.npair(v)
    if o < 3 or v < 3:
        if terms is True or terms == 'gradient':
            x = _Sums(o, v, True)
            out = (x.l1, np.ascontiguousarray(x.lt.T), x.goo, x.gvv, x.dvo)
            if terms is True:
                return 0.0, out
            return 0.0, out, Densities(np.zeros((o, v, npv), dtype=np.complex128), x.g2,
                                       np.ascontiguousarray(x.g3.T))
        return 0.0
    mo_e = np.asarray(eris.mo_energy, dtype=np.float64)
    eo, ev = np.ascontiguousarray(mo_e[:o]), np.ascontiguousarray(mo_e[o:])
    fvo = np.ascontiguousarray(np.asarray(eris.fock)[o:, :o], dtype=np.complex128)
    io, jo = K.tril(o)
    budget = memory.budget(mycc.max_memory if max_memory is None else max_memory)
    if three_m is None:
        # the real parts, if both operands fit in memory with them next to a
        # whole batch: an operand in a scratch file or small batches cost
        # more than the three real GEMMs save
        three_m = budget is None or budget >= (
            memory_need(o, v, True)[0] + 24 * (npv * o + v * npo) * (v + o)
            + max(memory_need(o, v, True)[1], BATCH_BYTES))
    m3 = bool(three_m)
    nparts, dtype, xbytes = (3, np.float64, 24) if m3 else (1, np.complex128, 16)
    if kramers is None:
        kramers = bool(getattr(eris, 'kramers', False))
        if kramers:
            from socutils.cc.fast.kramers import symmetric
            if not symmetric(t1, t2):
                log.info('(T): the amplitudes do not have the symmetry of the Kramers pairs: '
                         'all triples are done')
                kramers = False
    hv = v // 2 if kramers else v                   # b runs over the unbarred virtuals
    den = isinstance(terms, str) and terms == 'gradient'
    lam = terms is True or den
    if lam:
        terms = None
    elif terms is not None:
        terms.kramers = bool(kramers)

    # where the operands live: T2pT and VpT in memory; A and B in memory, the
    # smaller first, while they fit next to the smallest batch
    fixed, floor = memory_need(o, v, m3, 'gradient' if den else lam)
    n = v + o                                       # the joint index (e | m)
    shapes = {'A': (npv, o, n), 'B': (v, npo, n)}
    if budget is not None and budget < fixed + floor:
        raise MemoryError(
            '(T) needs %.1f MB (three packed T2 and the batch of one virtual orbital), %.1f MB '
            'are left of max_memory' % ((fixed + floor) / 1e6, budget / 1e6))
    left = None if budget is None else budget - fixed - floor
    ops, scr = {}, None
    for name in sorted(shapes, key=lambda k: np.prod(shapes[k], dtype=np.int64)):
        size = xbytes * int(np.prod(shapes[name], dtype=np.int64))
        if left is None or size <= left:
            ops[name] = tuple(np.empty(shapes[name], dtype=dtype) for _ in range(nparts))
            left = None if left is None else left - size
        else:
            if scr is None:
                scr = scratch.Scratch(prefix='zccsd_t_', what='the arrays of (T)',
                                      hint=' or give the calculation more memory (max_memory)')
            ops[name] = tuple(scr.create('%s%d' % (name, k), shapes[name], dtype)
                              for k in range(nparts))
    # the density of <bc||ek> as the batches add to it, [(b>c), k, e] like the
    # rows of A: after the operands
    g1 = None
    if den:
        size = 16 * npv * o * v
        if left is None or size <= left:
            g1 = np.zeros((npv, o, v), dtype=np.complex128)
            left = None if left is None else left - size
        else:
            if scr is None:
                scr = scratch.Scratch(prefix='zccsd_t_', what='the arrays of (T)',
                                      hint=' or give the calculation more memory (max_memory)')
            g1 = scr.create('g1', (npv, o, v), np.complex128)
    # the products of the terms of Lambda leave GEMM in an array of their own:
    # the least (one virtual, one pair), and an eighth of what there is more
    wa, wb = (n, n) if den else (v, o)      # of A and of B: the columns of these products
    rbytes = xbytes * max(wa * npo, o * wb) if lam else 0
    if left is None:
        rbytes = max(rbytes, REVERSE_BYTES) if lam else 0
        nbytes = BATCH_BYTES
    else:
        rbytes += min(left // 8, max(0, REVERSE_BYTES - rbytes)) if lam else 0
        # not all of what is left: the process grows by some 20 MB during the
        # pass that are no array of this plan (benzene/cc-pVDZ with a frozen
        # core under 4000 MB: a peak of 4.02 GB when the batches took it all)
        nbytes = min(BATCH_BYTES, floor + left - rbytes - min(left // 16, MARGIN_BYTES))
    rcap = rbytes // xbytes
    # the batches: (a, a1, None), a range of a with all its b - b < a, or with
    # Kramers pairs b below the barred half -, or (a, a + 1, (b0, b1)), one a
    # and a range of b; cap: the elements of the x of the largest batch
    plan, cap, a = [], 0, 2
    while a < v:
        if a < hv:
            a1 = _a_range(a, hv, o, nbytes, xbytes)
            size = K.npair(a1 - 1) * (a1 - a) + (K.npair(a1) - K.npair(a)) * (a1 - 1)
        else:       # barred a: x_{a,(bc)} of the range and x_{y,(ax)} of one a at a time
            a1 = a + max(0, min(v - a, (nbytes // (xbytes * o * npo) - hv * hv) // K.npair(hv)))
            size = K.npair(hv) * (a1 - a) + hv * hv
        if a1 > a:
            plan.append((a, a1, None))
            cap = max(cap, size)
            a = a1
        else:
            for b0, b1 in _batches(min(a, hv), o, nbytes, xbytes):
                plan.append((a, a + 1, (b0, b1)))
                cap = max(cap, K.npair(b1) - K.npair(b0) + (b1 - b0) * (b1 + b0))
            a += 1
    cap *= o * npo
    mapped = sorted(k for k in ops if scratch.is_mapped(ops[k][0]))
    log.info('(T): %d occupied and %d virtual spinors%s; %s; %d batches of at most %.2f GB; %s',
             o, v, ' in Kramers pairs (half of the virtual triples)' if kramers else '',
             'in a scratch file: ' + ', '.join(
                 '%s %.2f GB' % (k, xbytes * np.prod(shapes[k], dtype=np.int64) / 1e9)
                 for k in mapped) if mapped else 'all arrays in memory',
             len(plan), xbytes * cap / 1e9, 'three real GEMMs per product' if m3 else 'ZGEMM')
    if lam:
        log.info('(T): with the terms of the Lambda equations and of the density; their '
                 'products in blocks of %.1f MB', rbytes / 1e6)
    if den:
        log.info('(T): with the densities of the orbital gradient; that of <bc||ek>, %.2f GB, '
                 '%s', 16e-9 * npv * o * v,
                 'in a scratch file' if scratch.is_mapped(g1) else 'in memory')
    log.info('(T): matrix products on %s', blas.backend()[1])

    def blocks(nrow, per_row):
        '''Row ranges of an operand: all at once in memory, else within nbytes.'''
        step = nrow if scr is None else max(1, min(nrow, nbytes // per_row))
        return [(r, min(r + step, nrow)) for r in range(0, nrow, step)]

    def done(out):
        for x in out:
            scratch.written(x)
            scratch.unmap(x)

    with timer('t.setup'):
        t2 = np.ascontiguousarray(t2, dtype=np.complex128)
        t2p = K.pack_t2(t2) if t2.ndim == 4 else t2
        T2pT = np.ascontiguousarray(t2p.T)                      # [(b>c), (j>k)]
        oovv_p = getattr(eris, 'oovv_p', None)
        if oovv_p is None:
            oovv_p = K.pack_t2(np.ascontiguousarray(np.asarray(eris.oovv), dtype=np.complex128))
        VpT = np.ascontiguousarray(np.asarray(oovv_p).conj().T)     # <bc||jk>
        # B[a, (j>k), (e | m)] = (t2[j,k,a,e] | -<ma||jk>), <ma||jk> = <jk||ma>*
        B = ops['B']
        ooov_p = np.asarray(eris.ooov)[io, jo]                  # [(j>k), m, a]
        for a0, a1 in blocks(v, xbytes * npo * n):
            out = [x[a0:a1] for x in B]
            if not _put('zc_t_t2u', K._c(t2p), (npo, v, a0, a1, n, 0), out, m3):
                _store(K.unpack_tril(t2p, v, 'apb')[a0:a1], [x[:, :, :v] for x in out])
            _store(-ooov_p[:, :, a0:a1].transpose(2, 0, 1).conj(), [x[:, :, v:] for x in out])
            done(out)
        # A[(b>c), i, (e | m)] = (<bc||ei> | t2[i,m,b,c]), <bc||ei> = -<ie||bc>* from
        # the integrals as the CCSD keeps them, [i, e, (b>c)]: blocks of pairs
        A = ops['A']
        ovvv_p = getattr(eris, 'ovvv_p', None)
        if ovvv_p is None:
            bv, cv = K.tril(v)
            ovvv_p = np.ascontiguousarray(np.asarray(eris.ovvv)[:, :, bv, cv],
                                          dtype=np.complex128)
        for q0, q1 in blocks(npv, xbytes * o * n):
            out = [x[q0:q1] for x in A]
            if not _put('zc_t_t2x', K._c(t2p), (o, npv, q0, q1, n, v), out, m3):
                _store(K.unpack_tril(np.ascontiguousarray(T2pT[q0:q1]), o, 'pab'),
                       [x[:, :, v:] for x in out])
            for i in range(o):
                src = ovvv_p[i]
                if not (src.dtype == np.complex128 and src.flags.c_contiguous and _put(
                        'zc_t_ovvv', src.ctypes.data + 16 * q0, (npv, v, q1 - q0, o, i, n, 0),
                        out, m3)):
                    _store(-np.asarray(src[:, q0:q1]).T.conj(), [x[:, i, :v] for x in out])
                scratch.unmap(src)
            done(out)
        del t2p, ooov_p

    # the x of all batches are written into the same arrays: arrays allocated
    # per batch are mapped and zeroed page by page every time, as much memory
    # traffic as the kernel's own
    pool = [np.empty(cap, dtype=dtype) for _ in range(nparts)]
    sums = rbuf = None
    if lam:
        sums = _Sums(o, v, den)
        sums.g1 = g1
        rbuf = [np.empty(rcap, dtype=dtype) for _ in range(nparts)]

    def products(rows, cols, off=0):
        '''x = A[rows] B[cols]^T over the joint index (e | m), for a block of
        rows (pairs, i) and of columns (virtual, jk): complex, or its three
        real products, in the arrays of the batch from element ``off`` on.
        Returns x and the element behind it.'''
        a, b = [rows(x) for x in A], [cols(x) for x in B]
        m, nc = a[0].shape[0], b[0].shape[1]
        end = off + m * nc
        assert end <= cap
        out = [blas.gemm(a[k], b[k], pool[k][off:end].reshape(m, nc)) for k in range(nparts)]
        return (out if m3 else out[0]), end

    def reverse(X, r0, r1, s0, s1):
        '''The two products of the terms of Lambda for x[(r, i), (s, (jk))], the
        pairs r0 <= r < r1 of A times the virtuals s0 <= s < s1 of B, after the
        kernel has put the triples M in its place: Q[e, (s, (ij))], the sum
        over (r, k) of A[r, k, e] M, and Y[(r, i), j], the sum over
        (s, (mn)) of M B[s, (mn), v + j], antisymmetrized into sums.lt.  With
        the densities both run over all of A and of B: the rows of Q for
        A[r, k, v + j] = t2[k,j,r] are minus the density of <js||mn>, the
        columns of Y for B[s, (mn), e] = t2[m,n,s,e] the density of <r||ei>.'''
        X = X if m3 else (X,)
        nr, ns = (r1 - r0) * o, (s1 - s0) * npo
        assert X[0].shape == (nr, ns)
        ptr = lambda x: [K._c(y, dtype) for y in x] + [None, None]  # noqa: E731
        step = max(1, rcap // (wa * npo))
        for sa in range(s0, s1, step):
            sb = min(s1, sa + step)
            nc = (sb - sa) * npo
            with timer('t.lam.gemm'):
                Q = [blas.gemm(A[k][r0:r1].reshape(nr, n)[:, :wa].T,
                               X[k][:, (sa - s0) * npo:(sb - s0) * npo],
                               rbuf[k][:wa * nc].reshape(wa, nc)) for k in range(nparts)]
            with timer('t.lam.add'):
                if _HAVE_LAM and K.USE_C:
                    if den:
                        K._lib.zc_t_den_hole(*ptr(Q)[:3], int(m3), nc, v, o, npo, sa, sb,
                                             K._c(sums.g2))
                    K._lib.zc_t_lam_particle(*ptr(Q)[:3], int(m3), nc, v, npo, sa, sb,
                                             K._c(sums.lt))
                    continue
                if den:
                    sums.g2[:, sa:sb] -= _complex([q[v:] for q in Q] if m3 else Q[0][v:]).reshape(
                        o, sb - sa, npo)
                Q = _complex([q[:v] for q in Q] if m3 else Q[0][:v]).reshape(v, sb - sa, npo)
                for x in range(sa, sb):
                    sums.lt[K.npair(x):K.npair(x) + x] += Q[:x, x - sa]
                    e = np.arange(x + 1, v)
                    sums.lt[e * (e - 1) // 2 + x] -= Q[x + 1:, x - sa]
        step = max(1, rcap // (o * wb))
        for ra in range(r0, r1, step):
            rb = min(r1, ra + step)
            rows = (rb - ra) * o
            with timer('t.lam.gemm'):
                Y = [blas.gemm(X[k][(ra - r0) * o:(rb - r0) * o],
                               B[k][s0:s1].reshape(ns, n)[:, n - wb:],
                               rbuf[k][:rows * wb].reshape(rows, wb)) for k in range(nparts)]
            with timer('t.lam.add'):
                if _HAVE_LAM and K.USE_C:
                    K._lib.zc_t_lam_hole(*ptr(Y)[:3], int(m3), o, wb, wb - o, ra, rb,
                                         K._c(sums.lt))
                    if den:
                        K._lib.zc_t_den_rows(*ptr(Y)[:3], int(m3), rows, wb, v,
                                             K._c(sums.g1[ra:rb]))
                else:
                    Y = _complex(Y if m3 else Y[0]).reshape(rb - ra, o, wb)
                    Yh = Y[:, :, wb - o:]
                    sums.lt[ra:rb] += (Yh - Yh.transpose(0, 2, 1))[:, io, jo]
                    if den:
                        sums.g1[ra:rb] += Y[:, :, :v]
                if den:
                    scratch.written(sums.g1[ra:rb])

    def block(X, rows, cols):
        '''The rows and columns of the x of a product that one a uses.'''
        return [x[rows, cols] for x in X] if m3 else X[rows, cols]

    def one_a(a, b0, b1):
        '''The triples of one a with b0 <= b < b1: x_{a,(bc)} for the pairs with
        b in the range, x_{y,(ax)} for x < b1, b0 <= y < b1 and for
        b0 <= x < b1, y < b0.'''
        pa, nb, p0, p1 = K.npair(a), b1 - b0, K.npair(b0), K.npair(b1)
        with timer('t.gemm'):
            X1, end = products(lambda x: x[p0:p1].reshape((p1 - p0) * o, n), lambda x: x[a].T)
            G1, end = products(lambda x: x[pa:pa + b1].reshape(b1 * o, n),
                               lambda x: x[b0:b1].reshape(nb * npo, n).T, end)
            G2 = None
            if b0:
                G2, end = products(lambda x: x[pa + b0:pa + b1].reshape(nb * o, n),
                                   lambda x: x[:b0].reshape(b0 * npo, n).T, end)
        with timer('t.kernel'):
            e = _t_batch(X1, 0, G1, 0, G2, a, b0, b1, t1, fvo, VpT, T2pT, eo, ev, terms, sums)
        if lam:
            reverse(X1, p0, p1, a, a + 1)
            reverse(G1, pa, pa + b1, b0, b1)
            if b0:
                reverse(G2, pa + b0, pa + b1, 0, b0)
        return e

    et = 0j
    ntri = lambda a: K.npair(min(a, hv))                        # noqa: E731
    ndone, total, nxt = 0, sum(ntri(a) for a in range(v)), 0.1
    for a, a1, brange in plan:
        if brange is None and a >= hv:
            # barred a, unbarred b and c: x_{a,(bc)} as [(bc, i), (a, jk)] for the
            # pairs below the barred half, x_{y,(ax)} as [(x, i), (y, jk)] for one
            # a at a time (its pairs (a, x) are not next to those of a + 1)
            na, pr = a1 - a, K.npair(hv)
            with timer('t.gemm'):
                X1, end = products(lambda x: x[:pr].reshape(pr * o, n),
                                   lambda x: x[a:a1].reshape(na * npo, n).T)
            for aa in range(a, a1):
                pa = K.npair(aa)
                with timer('t.gemm'):
                    G1, _ = products(lambda x: x[pa:pa + hv].reshape(hv * o, n),
                                     lambda x: x[:hv].reshape(hv * npo, n).T, end)
                with timer('t.kernel'):
                    et += _t_batch(X1, (aa - a) * npo, G1, 0, None, aa, 0, hv,
                                   t1, fvo, VpT, T2pT, eo, ev, terms, sums)
                if lam:
                    reverse(G1, pa, pa + hv, 0, hv)
            if lam:
                reverse(X1, 0, pr, a, a1)
            ndone += na * ntri(a)
        elif brange is None:
            # x_{a,(bc)} as [(bc, i), (a, jk)] for the pairs below a1 - 1,
            # x_{y,(ax)} as [(a, x, i), (y, jk)] for y < a1 - 1
            na, pr, q0, q1 = a1 - a, K.npair(a1 - 1), K.npair(a), K.npair(a1)
            with timer('t.gemm'):
                X1, end = products(lambda x: x[:pr].reshape(pr * o, n),
                                   lambda x: x[a:a1].reshape(na * npo, n).T)
                G1, end = products(lambda x: x[q0:q1].reshape((q1 - q0) * o, n),
                                   lambda x: x[:a1 - 1].reshape((a1 - 1) * npo, n).T, end)
            with timer('t.kernel'):
                for aa in range(a, a1):
                    et += _t_batch(X1, (aa - a) * npo, G1, (K.npair(aa) - q0) * o, None, aa, 0, aa,
                                   t1, fvo, VpT, T2pT, eo, ev, terms, sums)
            for aa in range(a, a1) if lam else ():
                # of the x of the range, a has those of its pairs b < a and of y < a
                pa, col = K.npair(aa), (aa - a) * npo
                reverse(block(X1, slice(0, pa * o), slice(col, col + npo)), 0, pa, aa, aa + 1)
                reverse(block(G1, slice((pa - q0) * o, (pa + aa - q0) * o), slice(0, aa * npo)),
                        pa, pa + aa, 0, aa)
            ndone += sum(ntri(aa) for aa in range(a, a1))
        else:
            et += one_a(a, *brange)
            if brange[1] == min(a, hv):
                ndone += ntri(a)
        if ndone >= nxt * total:
            log.info('(T): %3.0f %% of the virtual triples, %.1f s', 100 * ndone / total,
                     logger.perf_counter() - cput0[1])
            nxt = (int(10 * ndone / total) + 1) / 10
    if kramers:
        et = 2.0 * et.real + 0j
    if abs(et.imag) > 1e-6:
        log.warn('Non-zero imaginary part found in the (T) energy %s', et)
    # the operands and the arrays of the batches go, with the views of them
    # that the loops left behind
    del A, B, ops, pool, rbuf
    X1 = G1 = out = None                                            # noqa: F841
    dens = None
    if lam:
        del T2pT, VpT           # two of the packed arrays: for the sums in their final layout
    if den:
        with timer('t.den.sort'):
            dens = _densities(mycc, sums, kramers, max_memory, log)
    del g1
    if scr is not None:
        scr.close()
    if lam:
        with timer('t.lam.add'):
            # lambda2 in its packed layout; the half of the sums that the
            # triples with a barred b would give
            l2p = np.ascontiguousarray(sums.lt.T)
            sums.lt = None
            out = [sums.l1, l2p, sums.goo, sums.gvv, sums.dvo]
            if kramers:
                from socutils.cc.fast.kramers import image, pair_maps, partner_signs
                K.ksym_packed(l2p, *pair_maps(o), *pair_maps(v))
                for k in (0, 4):
                    out[k] = out[k] + image(out[k])
                for k in (2, 3):
                    out[k] = out[k] + out[k][partner_signs(out[k].size)[0]].conj()
    log.timer('%s(T)' % mycc.__class__.__name__, *cput0)
    log.note('%s(T) correction = %.15g', mycc.__class__.__name__, et.real)
    if den:
        return et.real, tuple(out), dens
    return (et.real, tuple(out)) if lam else et.real


def _densities(mycc, sums, kramers, max_memory, log):
    '''The Densities of kernel() from its sums: the density of <bc||ek> from
    the layout of the batches, [(b>c), k, e], into that of the stored
    integrals, [k, e, (b>c)] - in memory if it takes no more than a quarter
    of what max_memory leaves now, else a mapped array of a scratch file of
    its own -, that of <bc||jk> transposed into the packed layout; with
    Kramers pairs all three completed by their time-reversed image (the
    triples that were done gave half of each).'''
    g1a, g2, g3 = sums.g1, sums.g2, sums.g3
    npv, o, v = g1a.shape
    budget = memory.budget(mycc.max_memory if max_memory is None else max_memory)
    scr = None
    if budget is None or 16 * g1a.size <= budget // 4:
        g1 = np.empty((o, v, npv), dtype=np.complex128)
    else:
        scr = scratch.Scratch(prefix='zccsd_td_', what='the density of (T)',
                              hint=' or give the calculation more memory (max_memory)')
        g1 = scr.create('g1', (o, v, npv), np.complex128)
        log.info('(T): the density of <bc||ek>, %.2f GB, in a scratch file', 16e-9 * g1a.size)
    step = max(1, min(npv, (64 << 20) // (16 * o * v)))
    for q0 in range(0, npv, step):
        q1 = min(npv, q0 + step)
        if _HAVE_LAM and K.USE_C:
            K._lib.zc_t_den_sort(K._c(g1a[q0:q1]), npv, o, v, q0, q1, K._c(g1))
        else:
            g1[:, :, q0:q1] = np.asarray(g1a[q0:q1]).transpose(1, 2, 0)
        scratch.unmap(g1a[q0:q1])
    sums.g1 = sums.g3 = None
    g3 = np.ascontiguousarray(g3.T)
    if kramers:
        from socutils.cc.fast.kramers import image, pair_maps, partner_signs
        g2 = g2 + image(g2, pairs=(2,))
        K.ksym_packed(g3, *pair_maps(o), *pair_maps(v))
        # g1 on its rows (k, e): the partner is (Kk, Ke), the sign that of the two
        (ko, so), (kv, sv) = partner_signs(o), partner_signs(v)
        K.ksym_packed(g1.reshape(o * v, npv), (ko[:, None] * v + kv[None, :]).ravel(),
                      (so[:, None] * sv[None, :]).ravel(), *pair_maps(v))
    scratch.written(g1)
    if scr is not None:
        # memory is short: the other two wait in the scratch file as well (the
        # Lambda equations are solved before they are used)
        for name, x in (('g2', g2), ('g3', g3)):
            y = scr.create(name, x.shape, np.complex128)
            y[...] = x
            scratch.written(y)
            if name == 'g2':
                g2 = y
            else:
                g3 = y
    return Densities(g1, g2, g3, scr)

#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Two-component (spinor / GHF) J and K from a real three-index factor B^P_{mu nu}
(pyscf df.DF in core or HDF5, socutils.cd.CD Cholesky vectors), (mu nu|ls) ~
sum_P B^P_{mu nu} B^P_{ls}, one pass over the factor for every orbital set and
density:

    K^{st}_{mu nu} = sum_P sum_i (B^P c^s_i)_mu (B^P c^t_i)^*_nu    s, t in {alpha, beta}
    J_{mu nu}      = sum_P B^P_{mu nu} sum_{ls} B^P_{ls} D_{ls}

for any set of (weighted) two-component orbitals c_i in the spherical GHF basis
(alpha rows first): the occupied orbitals of an SCF (weights sqrt(n_i)), the
frozen core of a CASSCF, or the active natural orbitals weighted by the square
root of their occupation (the active Fock matrix).

Each orbital gives four real AO vectors (Re / Im of the alpha and beta
components).  With the coefficients of a set as a real (nao, 4 ncol) matrix,
columns (i, component), the half transformation X^P = C^T B^P is one real GEMM
per P (pyscf's AO2MOnr_e2_drv on the packed lower triangle), and its output
(nb, ncol, 4, nao) is in memory the (nb ncol, 4 nao) matrix whose Gram matrix

    G_{x mu, y nu} = sum_{P i} X^P_{i x mu} X^P_{i y nu}       (x, y in aR, aI, bR, bI)

is a single rank-k update (dsyrk, 4 nao x 4 nao).  All exchange blocks follow
from G (K_aa = G_RR + G_II + i (G_IR - G_RI), etc.).  For an occupied set made
of Kramers pairs (u, Theta u) only the unbarred half is transformed and
K_aa = A + B^*, K_bb = B + A^*, K_ab = C - C^T from the unbarred sums A, B, C
(scf.spinor_hf._kramers_unbarred).

Flops per P, ncol transformed orbitals: half transformation 8 nao^2 ncol, Gram
matrix 16 nao^2 ncol (the 4 nao x 4 nao dsyrk); the exchange matrix of a
two-component set is the quaternion/complex analogue of a real K with 4 ncol
orbitals.
'''

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from socutils.scf import rik

BLK_MEMORY = .4       # fraction of the free memory for the block buffers
MAX_ROWS = 1 << 16    # rows (P, i) of the Gram update per block: enough for the dsyrk efficiency
QUAT = True           # Kramers sets by the four-product scheme (libsocvhf), else the Gram matrix
TILE = None           # orbitals per dgemm tile of the libsocvhf half transformation (None: all, up to
                      # a 64 MB tile per thread; one dgemm per P is the fastest)

# Kramers exchange from four products M_r = Y_{a_r}^T Y_{b_r} of combinations Y_v = sum_x v_x X_x of
# the half-transformed components X_x (x = aR, aI, bR, bI) instead of the 16 blocks G_xy = X_x^T X_y
# (10 of them distinct): with G's coefficient matrices, M_r = sum_xy a_rx b_ry G_xy, and
#     K0 = Re K_aa = sum_x G_xx                         = sum_r d_r (M_r + M_r^T)
#     Im K_aa = G10 - G01 - G32 + G23                    = sum_r c_1r (M_r - M_r^T)
#     Re K_ab = G02 + G13 - G20 - G31                    = sum_r c_2r (M_r - M_r^T)
#     Im K_ab = G12 - G03 - G21 + G30                    = sum_r c_3r (M_r - M_r^T)
# (K_bb = K_aa^*).  The quaternion structure of a Kramers-symmetric K (one symmetric, three
# antisymmetric real n x n parts) makes four full products of +-1 two-term combinations enough
# (an integer solution; at least four are needed since the three antisymmetric targets span an
# anti-self-dual subspace of the bivectors, which holds no decomposable a ^ b), i.e. 8 nao^2 per
# (P, orbital) instead of the Gram matrix's 16.
_QA = numpy.array([[0, 0, 1, -1], [0, 0, 1, 1], [0, 1, -1, 0], [1, -1, 0, 0]], dtype=float)
_QB = numpy.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, -1]], dtype=float)


def _quat_coefficients():
    e = numpy.eye(4)
    E = lambda a, b: numpy.outer(e[a], e[b])
    targets = [E(1, 0) - E(0, 1) - E(3, 2) + E(2, 3),
               E(0, 2) + E(1, 3) - E(2, 0) - E(3, 1),
               E(1, 2) - E(0, 3) - E(2, 1) + E(3, 0)]
    w = [numpy.outer(a, b) for a, b in zip(_QA, _QB)]
    A = numpy.array([(x - x.T).ravel() for x in w]).T
    S = numpy.array([(x + x.T).ravel() for x in w]).T
    c = numpy.array([numpy.linalg.lstsq(A, t.ravel(), rcond=None)[0] for t in targets])
    d = numpy.linalg.lstsq(S, e.ravel(), rcond=None)[0]
    assert abs(A @ c.T - numpy.array([t.ravel() for t in targets]).T).max() < 1e-12
    assert abs(S @ d - e.ravel()).max() < 1e-12
    return c.round(12), d.round(12)


_QC, _QD = _quat_coefficients()


def _comb_spec():
    '''(ia, ib, sb) of the eight two-term combinations a_1, b_1, a_2, b_2, ... for SOCVHFrik2c_half'''
    ia, ib, sb = [], [], []
    for r in range(4):
        for v in (_QA[r], _QB[r]):
            nz = numpy.flatnonzero(v)
            assert v[nz[0]] == 1
            ia.append(nz[0])
            ib.append(nz[1] if len(nz) > 1 else -1)
            sb.append(v[nz[1]] if len(nz) > 1 else 0.)
    return (numpy.array(ia, dtype=numpy.int32), numpy.array(ib, dtype=numpy.int32),
            numpy.array(sb, dtype=float))


def _tile(nao, ncol):
    if TILE:
        return TILE
    return int(min(ncol, max(128, (1 << 23) // (4 * nao))))


_libv = None


def _load_lib():
    global _libv
    if _libv is None:
        try:
            from socutils.lib import load_library
            lv = load_library('libsocvhf', 'SOCUTILS_SOCVHF_LIBRARY')
            _libv = lv if hasattr(lv, 'SOCVHFrik2c_half') else False
        except OSError:
            _libv = False
    return _libv


def half_combinations(eri1, c4, ncomb_spec, out=None, tile=None, dtril=None, vj=None):
    '''(ncomb, nb ncol, nao) two-term combinations of the half-transformed components (libsocvhf);
    with dtril (ndj, npair) (packed D + D^T, diagonal once), vj (ndj, npair) += the Coulomb
    matrices of these rows'''
    import ctypes
    if dtril is None:
        ndj, dptr, vptr = 0, None, None
    else:
        assert dtril.flags.c_contiguous and vj.flags.c_contiguous and vj.shape == dtril.shape
        ndj, dptr, vptr = dtril.shape[0], dtril.ctypes.data_as(ctypes.c_void_p), vj.ctypes.data_as(ctypes.c_void_p)
    ia, ib, sb = ncomb_spec
    eri1 = numpy.asarray(eri1, order='C')
    nb = eri1.shape[0]
    nao = c4.shape[0]
    ncol = c4.shape[1] // 4
    ncomb = len(ia)
    out = numpy.ndarray((ncomb, nb * ncol, nao), buffer=out)
    _load_lib().SOCVHFrik2c_half(
        out.ctypes.data_as(ctypes.c_void_p), eri1.ctypes.data_as(ctypes.c_void_p),
        c4.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nb), ctypes.c_int(nao), ctypes.c_int(ncol),
        ctypes.c_int(ncomb), ia.ctypes.data_as(ctypes.c_void_p), ib.ctypes.data_as(ctypes.c_void_p),
        sb.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(_tile(nao, ncol) if tile is None else tile),
        dptr, ctypes.c_int(ndj), vptr)
    return out


FUSED = True          # Kramers sets: half transformation + products per thread (SOCVHFrik2c_kquat)
FUSED_MEMORY = .25    # ... while the thread-private accumulators take at most this * the free memory


def kquat(acc_thr, eri1, c4, ncomb_spec, sign=1., dtril=None, vj=None, tile=None):
    '''acc_thr (nthreads, 4, nao, nao) += the four products of this factor block (libsocvhf,
    fused half transformation); with dtril, vj += its Coulomb matrices'''
    import ctypes
    ia, ib, sb = ncomb_spec
    eri1 = numpy.asarray(eri1, order='C')
    nb = eri1.shape[0]
    nao = c4.shape[0]
    ncol = c4.shape[1] // 4
    assert acc_thr.flags.c_contiguous and acc_thr.shape[0] >= lib.num_threads()
    if dtril is None:
        ndj, dptr, vptr = 0, None, None
    else:
        ndj, dptr, vptr = dtril.shape[0], dtril.ctypes.data_as(ctypes.c_void_p), vj.ctypes.data_as(ctypes.c_void_p)
    if tile is None:
        tile = TILE or int(min(ncol, max(256, (1 << 24) // (12 * nao))))
    _load_lib().SOCVHFrik2c_kquat(
        acc_thr.ctypes.data_as(ctypes.c_void_p), eri1.ctypes.data_as(ctypes.c_void_p),
        c4.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nb), ctypes.c_int(nao), ctypes.c_int(ncol),
        ia.ctypes.data_as(ctypes.c_void_p), ib.ctypes.data_as(ctypes.c_void_p), sb.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(tile), ctypes.c_double(sign), dptr, ctypes.c_int(ndj), vptr)
    return acc_thr


def _assemble_quat(ms, nao):
    '''the sph GHF exchange matrix of a Kramers set from the four products ms (4, nao, nao)'''
    s = [m + m.T for m in ms]
    a = [m - m.T for m in ms]
    k0 = sum(_QD[r] * s[r] for r in range(4))
    t = [sum(_QC[q, r] * a[r] for r in range(4)) for q in range(3)]
    k = numpy.empty((2 * nao, 2 * nao), dtype=complex)
    k[:nao, :nao] = k0 + 1j * t[0]
    k[nao:, nao:] = k[:nao, :nao].conj()
    k[:nao, nao:] = t[1] + 1j * t[2]
    k[nao:, :nao] = k[:nao, nao:].conj().T
    return k


def _real_columns(c, nao):
    '''(nao, 4 ncol) F-ordered real coefficients, columns (i, x) with x = aR, aI, bR, bI, of the
    two-component orbitals c (2 nao, ncol) in the spherical GHF basis'''
    ncol = c.shape[1]
    c4 = numpy.empty((nao, ncol, 4))
    c4[:, :, 0] = c[:nao].real
    c4[:, :, 1] = c[:nao].imag
    c4[:, :, 2] = c[nao:].real
    c4[:, :, 3] = c[nao:].imag
    return numpy.asarray(c4.reshape(nao, 4 * ncol), order='F')


def _assemble(g, nao, kramers):
    '''the sph GHF exchange matrix (2 nao, 2 nao) from the upper triangle of the Gram matrix g
    (4 nao, 4 nao), blocks ordered aR, aI, bR, bI'''
    u = numpy.triu(g)
    g = u + numpy.triu(u, 1).T
    blk = [[g[x * nao:(x + 1) * nao, y * nao:(y + 1) * nao] for y in range(4)] for x in range(4)]

    def kblk(s, t):
        # sum_i (s_R + i s_I)_mu (t_R - i t_I)_nu
        sr, si, tr, ti = 2 * s, 2 * s + 1, 2 * t, 2 * t + 1
        return (blk[sr][tr] + blk[si][ti]) + 1j * (blk[si][tr] - blk[sr][ti])
    a = kblk(0, 0)
    b = kblk(1, 1)
    c = kblk(0, 1)
    k = numpy.empty((2 * nao, 2 * nao), dtype=complex)
    if kramers:
        k[:nao, :nao] = a + b.conj()
        k[nao:, nao:] = b + a.conj()
        k[:nao, nao:] = c - c.T
    else:
        k[:nao, :nao] = a
        k[nao:, nao:] = b
        k[:nao, nao:] = c
    k[nao:, :nao] = k[:nao, nao:].conj().T
    return k


def get_jk(with_df, orbs=(), dms_j=(), kramers=None, signs=None, max_memory=None, verbose=None):
    '''J of real densities and two-component K of orbital sets, one pass over with_df's factor.

    orbs: list of (2 nao, ncol) complex orbital sets in the spherical GHF basis (alpha rows
    first), each column weighted (sqrt of its occupation).  kramers: list of bools, True when
    the set holds only the unbarred members of Kramers pairs (the barred partners Theta u are
    implied); default False.  signs: list of +1 / -1 (K of the set added with this sign;
    default +1).  dms_j: list of real (nao, nao) densities (the spin trace D_aa + D_bb).

    Returns (vj, vk): lists of real (nao, nao) Coulomb matrices and complex (2 nao, 2 nao) sph
    GHF exchange matrices K = sum_P B^P D B^P of D = sum_i c_i c_i^H (and its Kramers partner).'''
    log = logger.new_logger(with_df, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    if getattr(with_df, '_cderi', None) is None:
        with_df.build()
    nao = with_df.mol.nao_nr()
    npair = nao * (nao + 1) // 2
    naux = with_df.get_naoaux()
    orbs = [numpy.asarray(c) for c in orbs]
    nset = len(orbs)
    kramers = [False] * nset if kramers is None else list(kramers)
    signs = [1.] * nset if signs is None else [float(s) for s in signs]
    quat = [bool(kr) and QUAT and bool(_load_lib()) for kr in kramers]
    cols = [_real_columns(c, nao) if c.shape[1] > 0 else None for c in orbs]
    nthreads = lib.num_threads()
    if max_memory is None:
        max_memory = getattr(with_df, 'max_memory', 4000)
    mem_free = max(400., max_memory - lib.current_memory()[0] - 8 * nset * (4 * nao) ** 2 / 1e6)
    # fused Kramers sets: per-thread accumulators (and buffers), no shared half-transformed rows
    nq = sum(1 for c, q in zip(cols, quat) if q and c is not None)
    fused = FUSED and nq > 0 and 8 * nq * nthreads * 4 * nao * nao / 1e6 < FUSED_MEMORY * mem_free
    if fused:
        mem_free -= 8 * nq * nthreads * 4 * nao * nao / 1e6
    # doubles per auxiliary function and AO of the half-transformed rows: 4 components per orbital
    # (Gram matrix) or the 8 combinations (four-product scheme; none when fused)
    width = max([0] + [(0 if fused else 8 if q else 4) * (c.shape[1] // 4)
                       for c, q in zip(cols, quat) if c is not None])
    ncol_max = max([0] + [c.shape[1] // 4 for c, q in zip(cols, quat)
                          if c is not None and not (fused and q)])
    dms_j = [numpy.asarray(d, dtype=float) for d in dms_j]
    ndj = len(dms_j)

    # per auxiliary function: the factor block and its prefetch (2 npair), the half-transformed rows
    per_p = 8 * (2 * npair + width * nao) / 1e6
    blksize = int(max(1, min(naux, mem_free * BLK_MEMORY / per_p)))
    if ncol_max:
        blksize = min(blksize, max(1, MAX_ROWS // ncol_max))
    elif fused:
        blksize = min(blksize, 8 * nthreads)
    if blksize > nthreads:
        blksize -= blksize % nthreads      # the half transformation is parallel over P
    log.debug1('rik_2c.get_jk: naux %d, blksize %d, orbital sets %s (Kramers %s, four-product %s), '
               '%d J densities', naux, blksize, [0 if c is None else c.shape[1] // 4 for c in cols],
               kramers, quat, ndj)

    if ndj:
        d = numpy.array(dms_j)
        d = d + d.transpose(0, 2, 1)
        idx = numpy.arange(nao)
        d[:, idx, idx] *= .5
        dtril = numpy.ascontiguousarray(lib.pack_tril(d))  # sum over packed pairs k >= l of B_kl (D + D^T)_kl
        vj = numpy.zeros((ndj, npair))
    # the set whose libsocvhf half transformation also accumulates J (None: J by two GEMVs per block)
    jset = next((k for k in range(nset) if quat[k] and cols[k] is not None), None) if ndj else None
    acc = [(numpy.zeros((nthreads, 4, nao, nao)) if fused else numpy.zeros((4, nao, nao))) if q
           else numpy.zeros((4 * nao, 4 * nao), order='F') for q in quat]
    buf = numpy.empty(blksize * width * nao) if width else None
    spec = _comb_spec()
    tw = [0.] * 4          # wall: factor read, J, half transformation, Gram update / products
    tl = logger.perf_counter()
    for eri1 in rik.iter_factor(with_df, blksize):
        t1 = logger.perf_counter()
        tw[0] += t1 - tl
        assert eri1.shape[1] == npair
        if ndj and jset is None:
            rho = lib.dot(dtril, eri1.T)
            vj += lib.dot(rho, eri1)
        t2 = logger.perf_counter()
        tw[1] += t2 - t1
        for k in range(nset):
            if cols[k] is None:
                continue
            t3 = logger.perf_counter()
            if quat[k] and fused:
                kquat(acc[k], eri1, cols[k], spec, signs[k],
                      dtril=dtril if k == jset else None, vj=vj if k == jset else None)
                t4 = logger.perf_counter()
            elif quat[k]:
                # (8, nb ncol, nao); J of the densities with the first such set, from the same rows
                y = half_combinations(eri1, cols[k], spec, buf,
                                      dtril=dtril if k == jset else None, vj=vj if k == jset else None)
                t4 = logger.perf_counter()
                for r in range(4):
                    acc[k][r] += signs[k] * (y[2 * r].T @ y[2 * r + 1])   # (numpy: lib.dot is slow here)
            else:
                x = rik.half_transform(eri1, cols[k], buf)     # (nb, 4 ncol, nao) = (nb ncol, 4 nao)
                t4 = logger.perf_counter()
                acc[k] = rik._dsyrk_acc(x.reshape(-1, 4 * nao), acc[k], signs[k])
            tw[2] += t4 - t3
            tw[3] += logger.perf_counter() - t4
        eri1 = None
        tl = logger.perf_counter()
    log.debug1('rik_2c.get_jk wall: factor read %.2f s, J %.2f s, half transformation %.2f s, '
               'Gram update / products %.2f s', *tw)
    get_jk.last_timing = tw
    if fused:
        acc = [a.sum(axis=0) if q else a for a, q in zip(acc, quat)]
    vks = [_assemble_quat(a, nao) if q else _assemble(a, nao, kr) for a, q, kr in zip(acc, quat, kramers)]
    vjs = [lib.unpack_tril(v) for v in vj] if ndj else []
    log.timer_debug1('rik_2c.get_jk', *t0)
    return vjs, vks


def kramers_reduce(c, nao, tol=1e-8):
    '''Unbarred half u (2 nao, ncol / 2) of a weighted two-component orbital set c (2 nao, ncol) in
    the sph GHF basis whose density D = c c^H is time-reversal symmetric (Theta D Theta^-1 = D):
    u u^H + (Theta u) (Theta u)^H = D, the columns of c in any order or unitary mixture (natural
    orbitals of a Kramers-symmetric active 1-RDM weighted by the square root of their occupation,
    degenerate pairs mixed arbitrarily).  False when D is not time-reversal symmetric.

    Theta (c x) = c U x^* with the antisymmetric unitary U = c^+ Theta c; a symplectic Gram-Schmidt
    in the coefficient space C^ncol ({x_k, U x_k^*} orthonormal) gives u = c [x_1, x_2, ...].'''
    c = numpy.asarray(c)
    ncol = c.shape[1]
    if ncol == 0:
        return c
    if ncol % 2:
        return False
    tc = numpy.vstack((-c[nao:].conj(), c[:nao].conj()))
    u_mat = numpy.linalg.lstsq(c, tc, rcond=None)[0]
    scale = max(abs(c).max(), 1e-300)
    if abs(c @ u_mat - tc).max() > tol * scale or abs(u_mat + u_mat.T).max() > tol:
        return False
    xs = []
    basis = numpy.zeros((ncol, 0), dtype=complex)
    while basis.shape[1] < ncol:
        # the unit vector with the largest component outside the current span
        j = numpy.argmax(1 - numpy.einsum('jk,jk->j', basis.conj(), basis).real)
        x = numpy.zeros(ncol, dtype=complex)
        x[j] = 1
        x -= basis @ (basis.conj().T @ x)
        x -= basis @ (basis.conj().T @ x)
        x /= numpy.linalg.norm(x)
        xs.append(x)
        basis = numpy.hstack((basis, x[:, None], (u_mat @ x.conj())[:, None]))
    if basis.shape[1] != ncol:
        return False
    return c @ numpy.array(xs).T


def kramers_half(c, nao):
    '''unbarred half of c (2 nao, ncol) by the adjacent-pair check (scf.spinor_hf._kramers_unbarred,
    cheap) or else kramers_reduce; False when the set is not Kramers symmetric'''
    from socutils.scf.spinor_hf import _kramers_unbarred
    u = _kramers_unbarred(c, nao)
    if u is False:
        u = kramers_reduce(c, nao)
    return u


def get_jk_spinor(mol, with_df, mo_sets, with_j=True, kramers='auto', signs=None, max_memory=None,
                  verbose=None):
    '''J and K (spinor AO basis, (n2c, n2c)) of the densities D_s = C_s C_s^H of weighted spinor
    orbital sets C_s (n2c, ncol_s), one pass over with_df's factor (DF or CD):

        core Fock:    C = the frozen-core spinors                     (weights 1)
        active Fock:  C = active natural spinors * sqrt(occupation)   (diagonalised active 1-RDM)
        SCF:          C = occupied spinors * sqrt(mo_occ)

    kramers: 'auto' (a time-reversal symmetric set is reduced to its unbarred half: half the
    transformation, the four-product exchange), True (required) or False (general set).
    Returns (vj list or None, vk list).'''
    nao = mol.nao_nr()
    c2 = numpy.vstack(mol.sph2spinor_coeff())
    from socutils.scf.spinor_hf import sph2spinor
    orbs, krs = [], []
    for c in mo_sets:
        cs = c2 @ numpy.asarray(c)
        u = kramers_half(cs, nao) if kramers else False
        if kramers is True and u is False:
            raise ValueError('orbital set is not time-reversal symmetric')
        orbs.append(cs if u is False else u)
        krs.append(u is not False)
    dms = []
    if with_j:
        for c, kr in zip(orbs, krs):
            d = (lib.dot(c[:nao], c[:nao].conj().T) + lib.dot(c[nao:], c[nao:].conj().T)).real
            dms.append(2 * d if kr else d)
    vjs, vks = get_jk(with_df, orbs, dms, kramers=krs, signs=signs, max_memory=max_memory, verbose=verbose)
    vk = [sph2spinor(mol, k) for k in vks]
    if not with_j:
        return None, vk
    vj = []
    for j in vjs:
        js = numpy.zeros((2 * nao, 2 * nao), dtype=complex)
        js[:nao, :nao] = js[nao:, nao:] = j
        vj.append(sph2spinor(mol, js))
    return vj, vk

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
    cols = [_real_columns(c, nao) if c.shape[1] > 0 else None for c in orbs]
    ncol_max = max([0] + [c.shape[1] for c in cols if c is not None])
    dms_j = [numpy.asarray(d, dtype=float) for d in dms_j]
    ndj = len(dms_j)

    if max_memory is None:
        max_memory = getattr(with_df, 'max_memory', 4000)
    mem_free = max(400., max_memory - lib.current_memory()[0] - 8 * nset * (4 * nao) ** 2 / 1e6)
    # per auxiliary function: the factor block and its prefetch (2 npair), the half-transformed rows
    per_p = 8 * (2 * npair + ncol_max * nao) / 1e6
    blksize = int(max(1, min(naux, mem_free * BLK_MEMORY / per_p)))
    if ncol_max:
        blksize = min(blksize, max(1, MAX_ROWS * 4 // ncol_max))
    nthreads = lib.num_threads()
    if blksize > nthreads:
        blksize -= blksize % nthreads      # the half transformation is parallel over P
    log.debug1('rik_2c.get_jk: naux %d, blksize %d, orbital sets %s (Kramers %s), %d J densities',
               naux, blksize, [0 if c is None else c.shape[1] // 4 for c in cols], kramers, ndj)

    if ndj:
        d = numpy.array(dms_j)
        d = d + d.transpose(0, 2, 1)
        idx = numpy.arange(nao)
        d[:, idx, idx] *= .5
        dtril = lib.pack_tril(d)                     # sum over packed pairs k >= l of B_kl (D + D^T)_kl
        vj = numpy.zeros((ndj, npair))
    gs = [numpy.zeros((4 * nao, 4 * nao), order='F') for _ in range(nset)]
    buf = numpy.empty(blksize * ncol_max * nao) if ncol_max else None
    tw = [0.] * 4          # wall: factor read, J, half transformation, Gram update
    tl = logger.perf_counter()
    for eri1 in rik.iter_factor(with_df, blksize):
        t1 = logger.perf_counter()
        tw[0] += t1 - tl
        assert eri1.shape[1] == npair
        if ndj:
            rho = lib.dot(dtril, eri1.T)
            vj += lib.dot(rho, eri1)
        t2 = logger.perf_counter()
        tw[1] += t2 - t1
        for k in range(nset):
            if cols[k] is None:
                continue
            t3 = logger.perf_counter()
            x = rik.half_transform(eri1, cols[k], buf)     # (nb, 4 ncol, nao) = (nb ncol, 4 nao)
            t4 = logger.perf_counter()
            gs[k] = rik._dsyrk_acc(x.reshape(-1, 4 * nao), gs[k], signs[k])
            tw[2] += t4 - t3
            tw[3] += logger.perf_counter() - t4
        eri1 = None
        tl = logger.perf_counter()
    log.debug1('rik_2c.get_jk wall: factor read %.2f s, J %.2f s, half transformation %.2f s, '
               'Gram update %.2f s', *tw)
    get_jk.last_timing = tw
    vks = [_assemble(g, nao, kr) for g, kr in zip(gs, kramers)]
    vjs = [lib.unpack_tril(v) for v in vj] if ndj else []
    log.timer_debug1('rik_2c.get_jk', *t0)
    return vjs, vks

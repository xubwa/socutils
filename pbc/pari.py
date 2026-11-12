#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
# Pair-atomic resolution of the identity (PARI) at the Gamma point, on top of
# the periodic kernel of socutils.pbc.rsdf.
#
# Each lattice-summed AO pair rho_mn = sum_T mu_0 nu_T, with mu on atom A and
# nu on atom B, is fitted with the (lattice-summed) auxiliary functions of A
# and B only:
#     C^{mn} = M_dd^-1 (d|mn),  d = aux(A) u aux(B),
# M the metric of the G = 0-free periodic kernel.  Then
#     (mn|ls) ~ C^{mn} V C^{ls}                                  (PARI)
#     (mn|ls) ~ C^{mn} (.|ls) + (mn|.) C^{ls} - C^{mn} V C^{ls}    (robust)
# with V = M, (.|ls) the unfitted three-index integrals.  PARI is the local
# fit of Sodt/Subotnik/Head-Gordon (2006), Manzer et al. (2015, PARI-K) and
# Ihrig et al. (2015, RI-LVL in FHI-aims); the robust form is Dunlap's, studied
# for PARI by Merlot et al. (2013).
#
# This module is the accuracy prototype: it uses the full (aux|pair) tensor
# and metric of an RSDF built with keep_raw=True and dense intermediates, to
# decide which variant is accurate enough before the linear-scaling version
# (sparse C, local V_SR plus low-rank V_LR) is written.
#

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.df import addons as df_addons


def aux_atoms(cell, auxbasis):
    '''Atom index of each auxiliary function (RSDF's aux ordering).'''
    auxmol = df_addons.make_auxmol(cell, auxbasis)
    loc = auxmol.ao_loc_nr()
    return numpy.repeat([auxmol.bas_atom(i) for i in range(auxmol.nbas)], numpy.diff(loc))


def ao_atoms(cell):
    loc = cell.ao_loc_nr()
    return numpy.repeat([cell.bas_atom(i) for i in range(cell.nbas)], numpy.diff(loc))


def pari_coefficients(rsdf, lindep=1e-12, extra=None):
    '''Dense (naux, npair) PARI coefficients (zero outside each pair's domain).

    extra(A, B) -> additional atoms whose aux functions join the domain of the
    pair (A, B) (default: none).'''
    cell = rsdf.cell
    assert rsdf.int3c is not None and rsdf.metric is not None, 'build RSDF with keep_raw=True'
    paux = aux_atoms(cell, rsdf.auxbasis)
    pao = ao_atoms(cell)
    nao = cell.nao
    tril = numpy.tril_indices(nao)
    pa, pb = pao[tril[0]], pao[tril[1]]
    B = rsdf.int3c
    M = rsdf.metric
    C = numpy.zeros_like(B)
    for A in range(cell.natm):
        for Bt in range(A + 1):
            cols = numpy.where(((pa == A) & (pb == Bt)) | ((pa == Bt) & (pb == A)))[0]
            if cols.size == 0:
                continue
            atoms = {A, Bt}
            if extra is not None:
                atoms |= set(extra(A, Bt))
            d = numpy.where(numpy.isin(paux, list(atoms)))[0]
            w, v = scipy.linalg.eigh(M[numpy.ix_(d, d)])
            keep = w > lindep * w.max()
            x = (v[:, keep] / w[keep]) @ (v[:, keep].T @ B[numpy.ix_(d, cols)])
            C[numpy.ix_(d, cols)] = x
    return C


def _unpack(X):
    return lib.unpack_tril(X)


def jk_from_factors(left, V, right, dm, with_j=True):
    '''J and K of (mn|ls) = sum_PQ left[P, mn] V[P, Q] right[Q, ls] (packed
    (naux, npair) factors; V None means the identity, i.e. left . right).'''
    L = _unpack(left)
    R = _unpack(right)
    if V is not None:
        R = (V @ R.reshape(R.shape[0], -1)).reshape(R.shape)
    vj = None
    if with_j:
        rho = numpy.einsum('Pls,sl->P', R, dm)
        vj = numpy.einsum('P,Pmn->mn', rho, L)
    # K_mn = sum_ls (ml|ns) D_ls = sum_P L_P,ml D_ls R_P,ns
    vk = numpy.einsum('Pml,ls,Pns->mn', L, dm, R, optimize=True)
    return vj, vk


def pari_jk(rsdf, dm, C=None, robust=False, with_j=True):
    '''J/K with the PARI (or robust PARI) four-index integrals (dense prototype).'''
    if C is None:
        C = pari_coefficients(rsdf)
    V = rsdf.metric
    vj, vk = jk_from_factors(C, V, C, dm, with_j)
    if robust:
        B = rsdf.int3c
        j1, k1 = jk_from_factors(C, None, B, dm, with_j)
        j2, k2 = jk_from_factors(B, None, C, dm, with_j)
        vk = k1 + k2 - vk
        if with_j:
            vj = j1 + j2 - vj
    return vj, vk


# ----------------------------------------------------------------------
# Range-separated local exchange: PARI with the short-range (erfc) kernel only
# (local fit, local metric) plus the long-range exchange evaluated exactly in
# G space from analytic pair FTs, with the G = 0 part of the SR kernel.
#
#   K = K_SR[PARI, erfc kernel] + sum_{G != 0} (v_G / V) Re[rho(G) D rho(G)^+]
#       - pi/(w^2 V) S D S
#
# rho(G) the lattice-summed AO pair FTs; v_G the erf kernel (one vector of
# each +-G pair, weight 2), as in RSDF.build.

def pari_coefficients_sr(rsdf, lindep=1e-12, extra=None):
    '''PARI coefficients of the SR (erfc) kernel: the fit and metric are local.'''
    class _V:
        pass
    v = _V()
    v.cell, v.auxbasis = rsdf.cell, rsdf.auxbasis
    v.int3c, v.metric = rsdf.int3c_sr, rsdf.metric_sr
    return pari_coefficients(v, lindep, extra)


def lr_exchange(rsdf, dm, omega=None):
    '''Exact long-range exchange (erf kernel, G != 0) minus the G = 0 part of
    the SR kernel, from lattice-summed AO pair FTs (dense prototype).'''
    from socutils.pbc.coulomb_reference import reciprocal_vectors
    from socutils.pbc.rsdf import _gamma_ft_kernel
    cell = rsdf.cell
    omega = rsdf.omega if omega is None else omega
    a = cell.lattice_vectors()
    vol = abs(numpy.linalg.det(a))
    lnp = -numpy.log(rsdf.precision)
    G = reciprocal_vectors(a, 2 * omega * numpy.sqrt(lnp + numpy.log(4 * numpy.pi)))
    b = 2 * numpy.pi * numpy.linalg.inv(a).T
    m = numpy.rint(G @ numpy.linalg.inv(b)).astype(int)
    half = (m[:, 0] > 0) | ((m[:, 0] == 0) & ((m[:, 1] > 0) | ((m[:, 1] == 0) & (m[:, 2] > 0))))
    G = G[half]
    G2 = numpy.einsum('gx,gx->g', G, G)
    vG = 2 * 4 * numpy.pi * numpy.exp(-G2 / (4 * omega**2)) / G2
    kern = _gamma_ft_kernel(cell, rsdf.precision * 1e-2)
    nao = cell.nao
    vk = numpy.zeros((nao, nao))
    blk = 256
    for g0 in range(0, len(G), blk):
        F = kern(G[g0:g0 + blk], kptjs=numpy.zeros((1, 3)), shls_slice=(0, cell.nbas, 0, cell.nbas))[:, 0]
        rho = lib.unpack_tril(F[0]) + 1j * lib.unpack_tril(F[1])    # (ng, nao, nao), symmetric
        x = numpy.einsum('gml,ls->gms', rho, dm)
        vk += numpy.einsum('g,gms,gns->mn', vG[g0:g0 + blk] / vol, x, rho.conj()).real
    F0 = kern(numpy.zeros((1, 3)), kptjs=numpy.zeros((1, 3)), shls_slice=(0, cell.nbas, 0, cell.nbas))[:, 0]
    S = lib.unpack_tril(F0[0, 0])
    vk -= numpy.pi / (omega**2 * vol) * (S @ dm @ S)
    return vk


def rs_pari_k(rsdf, dm, C_sr=None, extra=None):
    '''K = PARI(SR) + exact LR - G=0 (dense prototype).'''
    if C_sr is None:
        C_sr = pari_coefficients_sr(rsdf, extra=extra)
    _, k_sr = jk_from_factors(C_sr, rsdf.metric_sr, C_sr, dm, with_j=False)
    return k_sr + lr_exchange(rsdf, dm)

#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
# Long-range part of a range-separated Gamma-point J/K builder.
#
# The Coulomb kernel is split as 1/r = erfc(w r)/r + erf(w r)/r.  The long-range
# erf part decays in reciprocal space as exp(-G^2/4w^2), so for moderate w only a
# few reciprocal vectors are needed, and compact (all-electron core) AO products
# are smooth there.  Everything is evaluated with analytic Fourier transforms of
# explicitly enumerated AO pairs (molecular libcint FT integrals), so the only
# truncations are the G cutoff and the AO-pair radius, both set explicitly.
#
# Gamma-point Bloch pair densities chi~_mu chi~_lam have cell Fourier components
#     P_{mu lam}(G) = (1/V) sum_T  int chi_mu(r) chi_lam(r - T) exp(-iGr) dr
# and, with v(G) = 4 pi exp(-G^2/4w^2)/G^2 (G = 0 dropped),
#     J_{mu nu} = V sum_G v(G) P_{mu nu}(G)^* rho(G),   rho(G) = sum D_{mu lam} P_{mu lam}(G)
#     K_{mu nu} = V sum_G v(G) [P(G)^* D P(G)]_{mu nu}
# (pyscf conventions: K_il = sum_jk (ij|kl) D_jk; tin-foil G = 0).
#

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.gto.ft_ao import ft_aopair
from socutils.pbc.coulomb_reference import (image_mole, reciprocal_vectors, cell_volume)


def lr_gcut(omega, precision):
    '''|G| beyond which 4 pi exp(-G^2/4w^2)/G^2 < precision (exp factor alone, conservative).'''
    return 2 * omega * np.sqrt(-np.log(precision) + np.log(4 * np.pi))


def pair_radius(cell, precision):
    '''Distance beyond which every AO-pair overlap is below precision: the
    Gaussian product of exponents a, b decays as exp(-ab/(a+b) r^2).'''
    amin = min(cell.bas_exp(i).min() for i in range(cell.nbas))
    return np.sqrt(-np.log(precision) / (amin / 2))


class LongRangeJK(lib.StreamObject):
    '''Long-range (erf) Gamma-point J and K for a cell.'''

    def __init__(self, cell, omega, precision=1e-10, rpair=None, max_memory=None):
        self.cell = cell
        self.omega = omega
        self.precision = precision
        self.rpair = rpair
        self.max_memory = max_memory or cell.max_memory
        self.verbose = cell.verbose
        self.stdout = cell.stdout
        self._mol = None
        self._fold = None
        self._G = None
        self._P = None  # cached P(G) when it fits in memory

    def build(self):
        cell = self.cell
        if self.rpair is None:
            self.rpair = pair_radius(cell, self.precision)
        self._mol, self._fold = image_mole(cell, self.rpair)
        G = reciprocal_vectors(cell.lattice_vectors(), lr_gcut(self.omega, self.precision))
        G2 = np.einsum('gx,gx->g', G, G)
        self._G = G
        self._vG = 4 * np.pi * np.exp(-G2 / (4 * self.omega**2)) / G2
        self._vol = cell_volume(cell.lattice_vectors())
        nao = cell.nao
        # fold matrix: image-mole AO -> cell AO
        self._M = np.zeros((self._mol.nao, nao))
        self._M[np.arange(self._mol.nao), self._fold] = 1.0
        mem = len(G) * nao**2 * 16 / 1e6
        logger.info(self, 'LongRangeJK: omega %.3f, %d G vectors, %d image AOs (rpair %.1f bohr), P(G) %.0f MB',
                    self.omega, len(G), self._mol.nao, self.rpair, mem)
        if mem < .5 * self.max_memory:
            self._P = np.concatenate([p for _, p in self._P_blocks(cache=False)])
        return self

    def _P_blocks(self, blksize=None, cache=True):
        '''Yield (slice, P(G) block) with P of shape (ng_blk, nao, nao).'''
        if cache and self._P is not None:
            yield slice(0, len(self._G)), self._P
            return
        mol = self._mol
        nao = self.cell.nao
        if blksize is None:
            blksize = max(16, int(self.max_memory * .1e6 / (16 * nao * mol.nao * 2)))
        shls_slice = (0, self.cell.nbas, 0, mol.nbas)
        for g0 in range(0, len(self._G), blksize):
            g1 = min(g0 + blksize, len(self._G))
            F = ft_aopair(mol, self._G[g0:g1], shls_slice=shls_slice)  # (ng, nao, nao_img)
            yield slice(g0, g1), (F @ self._M) / self._vol

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True):
        if self._G is None:
            self.build()
        dm = np.asarray(dm)
        nao = self.cell.nao
        dms = dm.reshape(-1, nao, nao)
        is_complex = np.iscomplexobj(dms)
        vj = np.zeros(dms.shape, dtype=complex) if with_j else None
        vk = np.zeros(dms.shape, dtype=complex) if with_k else None
        vol = self._vol
        for sl, P in self._P_blocks():
            v = self._vG[sl]
            Pc = P.conj()
            for i, d in enumerate(dms):
                if with_j:
                    rho = np.einsum('gij,ij->g', P, d)
                    vj[i] += vol * np.einsum('g,gij->ij', v * rho, Pc)
                if with_k:
                    # sum_g v_g conj(P_g) D P_g as one GEMM over the (g, lam) index
                    X = (Pc @ d) * v[:, None, None]                      # (ng, mu, sig)
                    vk[i] += vol * X.transpose(1, 0, 2).reshape(nao, -1) @ P.reshape(-1, nao)
        if not is_complex:
            vj = vj.real if with_j else None
            vk = vk.real if with_k else None
        if with_j:
            vj = vj.reshape(dm.shape)
        if with_k:
            vk = vk.reshape(dm.shape)
        return vj, vk

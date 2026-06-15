'''Correct incore spinor (j-spinor) AO->MO transform with s4 AO symmetry.

Computes the chemist MO integrals (ij|kl) over j-spinor MOs as

    g = Bbra . eri_s4 . Bket^T

where eri_s4 = mol.intor('int2e_sph', aosym='s4') (the spherical AO integrals
with i>=j, k>=l permutational symmetry -- ~4x fewer shell quartets than the
s1-only nrr_outcore path) and Bbra/Bket are the folded two-component
(alpha+beta) j-spinor transition densities packed over the s4 AO triangle:

    Bbra[ij, (m>=n)] = sum_spin (C*_mi C_nj + [m!=n] C*_ni C_mj)

so the contraction with the (single-stored) s4 AO pair is exact. Both
half-transforms are BLAS zgemm; no full nao^2 AO intermediate and no
transposes are formed.

Status: VERIFIED correct to machine precision vs int2e_spinor (blocks + full
transform). NOT faster than the stock C nrr_outcore at moderate sizes -- the
numpy/zgemm transform does not beat pyscf's C transform, and the s4 saving is
on AO generation, which is not the bottleneck here. A genuine speedup needs a
C-level s4 outcore (port nr's AO2MOtranse2_nr_s2kl + a complex mmm); see
NOTES.md.
'''
import numpy as np
from pyscf import lib


def _fold(a0, a1, b0, b1, nao):
    '''Folded two-component transition density, packed over the AO triangle.'''
    D = lib.einsum('mi,nj->ijmn', a0.conj(), a1)
    D += lib.einsum('mi,nj->ijmn', b0.conj(), b1)
    D = D.reshape(-1, nao, nao)
    M = D + D.transpose(0, 2, 1)
    ix = np.arange(nao)
    M[:, ix, ix] = D[:, ix, ix]          # undo the doubled diagonal
    return lib.pack_tril(M)              # (nij, npair)


def general_incore(mol, mo_coeffs, intor='int2e_sph', eri_s4=None):
    '''(ij|kl) chemist over four sets of j-spinor MOs (s4 AO integrals).'''
    ca, cb = mol.sph2spinor_coeff()
    nao = ca.shape[0]
    moa = [ca.dot(m) for m in mo_coeffs]
    mob = [cb.dot(m) for m in mo_coeffs]
    if eri_s4 is None:
        eri_s4 = mol.intor(intor, aosym='s4')
    Bbra = _fold(moa[0], moa[1], mob[0], mob[1], nao)
    Bket = _fold(moa[2], moa[3], mob[2], mob[3], nao)
    g = Bbra.dot(eri_s4).dot(Bket.T)
    nmoi, nmoj, nmok, nmol = [m.shape[1] for m in mo_coeffs]
    return g.reshape(nmoi, nmoj, nmok, nmol)


def general_iofree_incore(mol, mo_coeffs, intor='int2e_sph', eri_s4=None, **kw):
    g = general_incore(mol, mo_coeffs, intor, eri_s4)
    nmoi, nmoj, nmok, nmol = [m.shape[1] for m in mo_coeffs]
    return g.reshape(nmoi * nmoj, nmok * nmol)

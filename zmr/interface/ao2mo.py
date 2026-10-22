'''
In-core AO -> MO transformation of the Coulomb integrals for two-component
orbitals (j-spinors or GHF spin-orbitals).

The real spherical AO integrals are computed once with their permutational
symmetry within each pair (aosym = s4, nao (nao + 1) / 2 pairs) and kept in
memory.  For orbitals with alpha / beta spherical components (a, b),

    (ij|kl) = B_bra[(ij), (mn)] (mn|ls) B_ket[(kl), (ls)]
    B[(ij), (m >= n)] = sum_spin conj(c_mi) c_nj + [m != n] conj(c_ni) c_mj

so a block needs one folding step (OpenMP kernel) and two matrix products
(MKL).  This replaces pyscf.ao2mo.nrr_outcore, whose spinor path uses no AO
symmetry and writes through HDF5; that routine remains the fallback when the
AO integrals do not fit into memory.
'''

import numpy
from pyscf import lib as pyscf_lib

from socutils.zmr.lib import kernels
from socutils.zmr.lib import linalg as la


def _same(x, y):
    return x is y or (x.shape == y.shape and numpy.array_equal(x, y))


class IncoreAO2MO:
    '''eri_fn(c1, c2, c3, c4) -> (pq|rs) for two-component MO coefficients.

    Args:
        mol: PySCF Mole
        components: function mapping a coefficient block to its (alpha, beta)
            spherical AO components, each (nao, n)
        fallback: eri_fn used when the s4 AO integrals exceed max_memory (MB)
    '''

    def __init__(self, mol, components, fallback, max_memory=None):
        self.mol = mol
        self.components = components
        self.fallback = fallback
        self.max_memory = mol.max_memory if max_memory is None else max_memory
        self._eri = None
        nao = mol.nao_nr()
        self.npair = nao * (nao + 1) // 2
        self.incore = self.npair**2 * 8 / 1e6 < 0.5 * self.max_memory

    def eri_ao(self):
        if self._eri is None:
            self._eri = self.mol.intor('int2e_sph', aosym='s4')
        return self._eri

    def __call__(self, c1, c2, c3, c4):
        if not self.incore:
            return self.fallback(c1, c2, c3, c4)
        shape = tuple(c.shape[1] for c in (c1, c2, c3, c4))
        if 0 in shape:
            return numpy.zeros(shape, dtype=numpy.complex128)
        a1, b1 = self.components(c1)
        a2, b2 = self.components(c2)
        bra = kernels.fold_density(a1, b1, a2, b2)
        if _same(c3, c1) and _same(c4, c2):
            ket = bra
        else:
            a3, b3 = self.components(c3)
            a4, b4 = self.components(c4)
            ket = kernels.fold_density(a3, b3, a4, b4)
        eri = self.eri_ao()
        half = numpy.empty(bra.shape, dtype=numpy.complex128)
        # keep the intermediate products below ~1 GB
        step = max(1, int(1e9 // (16 * self.npair)))
        for p0, p1 in pyscf_lib.prange(0, bra.shape[0], step):
            half[p0:p1] = la.rgemm(bra[p0:p1].real, eri) + 1j * la.rgemm(bra[p0:p1].imag, eri)
        return la.gemm(half, ket, 'N', 'T').reshape(shape)

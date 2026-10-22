'''
Integrals of a non-relativistic Hamiltonian in real spatial orbitals with a
closed-shell (doubly occupied) core:

    H = ecore + sum_pq h_pq E_pq + 1/2 sum_pqrs (pq|rs) (E_pq E_rs - delta_qr E_ps),
    E_pq = sum_sigma p^+_sigma q_sigma.

Same interface as zmr.integrals (blocks by space letters, rotate,
generalized_fock, core_energy), with OrbitalSpaces counting spatial orbitals
and the active density given as the spin-summed 1-RDM D_tu = <E_tu>:

    f_pq = h_pq + sum_i [2 (pq|ii) - (pi|iq)] + sum_tu D_tu [(pq|tu) - 1/2 (pu|tq)]
    E_core = ecore + 2 sum_i h_ii + sum_ij [2 (ii|jj) - (ij|ji)]
'''

import numpy

from socutils.zmr import integrals


class ERIs(integrals.ERIsBase):
    '''Stored real integrals over spatial orbitals.'''

    dtype = numpy.float64

    def __init__(self, ecore, h1, eri, spaces):
        nmo = spaces.nmo
        self.ecore = float(ecore)
        self.h1 = numpy.asarray(h1, dtype=float).reshape(nmo, nmo)
        self._eri = numpy.asarray(eri, dtype=float).reshape((nmo,) * 4)
        self.spaces = spaces

    def eri_block(self, labels):
        if len(labels) != 4:
            raise ValueError('eri_block needs four space labels')
        return self._eri[tuple(self.spaces.slice(x) for x in labels)]

    def rotate(self, u):
        '''Hamiltonian in the orbitals phi'_p = sum_q phi_q u_qp for a real
        orthogonal u that respects the space partition.'''
        u = numpy.asarray(u, dtype=float)
        h1 = u.T @ self.h1 @ u
        eri = numpy.einsum('pqrs,pi,qj,rk,sl->ijkl', self._eri, u, u, u, u, optimize=True)
        return ERIs(self.ecore, h1, eri, self.spaces)

    def with_virtuals(self, u):
        '''As zmr.integrals.ERIs.with_virtuals, for a real u.'''
        full, spaces = self._virtual_map(u)
        h1 = full.T @ self.h1 @ full
        eri = numpy.einsum('pqrs,pi,qj,rk,sl->ijkl', self._eri, full, full, full, full,
                           optimize=True)
        return ERIs(self.ecore, h1, eri, spaces)

    def core_energy(self):
        c = self.spaces.slice('c')
        cccc = self.eri_block('cccc')
        return float(self.ecore + 2 * numpy.trace(self.h1[c, c])
                     + 2 * numpy.einsum('iijj->', cccc) - numpy.einsum('ijji->', cccc))

    def generalized_fock(self, dm_act):
        sp = self.spaces
        f = self.h1.copy()
        if sp.ncore:
            f += (2 * numpy.einsum('pqii->pq', self.eri_block('ppcc'))
                  - numpy.einsum('piiq->pq', self.eri_block('pccp')))
        if sp.nact:
            f += (numpy.einsum('pqtu,tu->pq', self.eri_block('ppaa'), dm_act)
                  - 0.5 * numpy.einsum('putq,tu->pq', self.eri_block('paap'), dm_act))
        return f


class MOERIs(integrals.MOERIs):
    '''Integral blocks transformed on demand from real MO coefficients.

    Args as zmr.integrals.MOERIs, with
        eri_fn(c1, c2, c3, c4) -> real (pq|rs)
        veff_fn(dm_ao) -> J[dm] - 1/2 K[dm] for a total (spin-summed) density
    '''

    dtype = numpy.float64
    _docc = 2
    _kact = 0.5

    def _dm(self, dm_act):
        '''Total AO density of the doubly occupied core plus the active
        spin-summed density.'''
        cc = self.coeff('c')
        ca = self.coeff('a')
        return 2 * cc @ cc.T + ca @ numpy.asarray(dm_act).T @ ca.T

    def core_energy(self):
        cc = self.coeff('c')
        if cc.shape[1] == 0:
            return self.ecore
        if self.factorized:
            return integrals.MOERIs.core_energy(self)
        dm = 2 * cc @ cc.T
        return float(self.ecore + numpy.einsum('ij,ji->', dm, self.hcore_ao)
                     + 0.5 * numpy.einsum('ij,ji->', dm, self.veff_fn(dm)))

    def to_dense(self):
        return ERIs(self.ecore, self.h1, self.eri_block('pppp'), self.spaces)

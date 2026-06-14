#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Spinor MP2 (two-component, j-adapted Kramers-unrestricted) on top of a
:class:`socutils.scf.spinor_hf.SpinorSCF` reference.

The spinor orbitals are full complex two-component functions, so every index
is a genuine spin-orbital and the working equations are the bare spin-orbital
MP2 expressions written with the antisymmetrised physicist integrals
``<ij||ab> = <ij|ab> - <ij|ba>``::

    t2_{ij}^{ab} = <ij||ab>* / (e_i + e_j - e_a - e_b)
    E_corr      = (1/4) sum_{ijab} <ij||ab> t2_{ij}^{ab}

The two-electron integrals are the spinor Coulomb integrals
``int2e_spinor`` (the (LL|LL) block); this matches the picture-change
convention used throughout socutils, where the spin-orbit physics lives in
the one-electron (X2C/X2CAMF) part of the mean field and the correlation
treatment consumes the bare spinor Coulomb integrals.  For a non-relativistic
:class:`SpinorSCF` reference the result is numerically identical to a real
restricted/generalised MP2 on the same Hamiltonian.

Orbital energies are taken from the mean field (``mf.mo_energy``).  Under a
legal canonical orbital rotation (per-orbital phases or a unitary mix within
an exactly degenerate -- e.g. Kramers -- energy block) the orbital energies
are unchanged and the diagonal-Fock structure the MP2 formula relies on is
preserved, so the correlation energy and ``||t2||`` are invariant while the
amplitudes themselves become complex.
'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=True):
    '''Spinor MP2 correlation energy (and, optionally, the t2 amplitudes).

    Returns ``(e_corr, t2)`` where ``t2`` has shape ``(nocc, nocc, nvir, nvir)``
    and is ``None`` when ``with_t2`` is False.
    '''
    if mo_energy is None:
        mo_energy = mp.mo_energy
    if mo_coeff is None:
        mo_coeff = mp.mo_coeff
    mo_energy = np.asarray(mo_energy)

    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc
    moidx = mp.get_frozen_mask()
    mo_energy = mo_energy[moidx]
    mo_coeff = mo_coeff[:, moidx]

    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:]

    # antisymmetrised physicist integrals <ij||ab>, shape (nocc,nocc,nvir,nvir)
    oovv = mp.ao2mo(mo_coeff) if eris is None else eris

    # D_{ijab} = e_i + e_j - e_a - e_b  (strictly negative for a HOMO-LUMO gap)
    d_ijab = (eo[:, None, None, None] + eo[None, :, None, None]
              - ev[None, None, :, None] - ev[None, None, None, :])

    t2 = oovv.conj() / d_ijab
    e_corr = 0.25 * np.einsum('ijab,ijab->', oovv, t2)
    e_corr = e_corr.real

    if not with_t2:
        t2 = None
    return e_corr, t2


class SpinorMP2(lib.StreamObject):
    '''Spinor MP2.

    Args:
        mf : a converged :class:`SpinorSCF` (or subclass) mean-field object.

    The MO coefficients/occupations are read from ``mf`` at construction time
    (or may be supplied explicitly), so injecting rotated orbitals -- e.g.
    ``mp = SpinorMP2(mf.copy().set(mo_coeff=C @ U))`` -- works without any
    re-diagonalisation: the same canonical ``mo_energy`` is reused.
    '''

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        self._scf = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory
        self.frozen = frozen

        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        # Canonical orbital energies: unchanged by a legal (phase /
        # degenerate-block) rotation, so they are reused verbatim.
        self.mo_energy = mf.mo_energy

        self.e_corr = None
        self.t2 = None

    @property
    def nocc(self):
        return int(np.count_nonzero(self.mo_occ > 0))

    @property
    def nmo(self):
        return self.mo_coeff.shape[1]

    @property
    def e_tot(self):
        return self.e_corr + self._scf.e_tot

    def get_frozen_mask(self):
        '''Boolean mask (length nmo) of orbitals kept in the correlation
        treatment.  ``frozen`` may be ``None``/0, an int (freeze the lowest
        ``frozen`` spinors) or an explicit list of frozen orbital indices.'''
        mask = np.ones(self.nmo, dtype=bool)
        frozen = self.frozen
        if frozen is None or (np.ndim(frozen) == 0 and frozen == 0):
            return mask
        if np.ndim(frozen) == 0:
            mask[:int(frozen)] = False
        else:
            mask[list(frozen)] = False
        return mask

    def ao2mo(self, mo_coeff=None):
        '''Antisymmetrised physicist integrals ``<ij||ab>`` over the active
        occupied/virtual spinor MOs, shape ``(nocc, nocc, nvir, nvir)``.'''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff[:, self.get_frozen_mask()]
        nocc = self.nocc
        orbo = mo_coeff[:, :nocc]
        orbv = mo_coeff[:, nocc:]

        # Chemist spinor Coulomb integrals (pq|rs) in the AO spinor basis.
        eri_ao = self.mol.intor('int2e_spinor')

        # Half-then-half transform to the chemist MO block (ia|jb).
        tmp = lib.einsum('pqrs,pi->iqrs', eri_ao, orbo.conj())
        tmp = lib.einsum('iqrs,qa->iars', tmp, orbv)
        tmp = lib.einsum('iars,rj->iajs', tmp, orbo.conj())
        ovov = lib.einsum('iajs,sb->iajb', tmp, orbv)

        # Physicist <ij|ab> = (ia|jb); antisymmetrise the virtual indices.
        phys = ovov.transpose(0, 2, 1, 3)
        oovv = phys - phys.transpose(0, 1, 3, 2)
        return oovv

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=True):
        log = logger.new_logger(self)
        if self.mo_coeff is None:
            raise RuntimeError('mo_coeff is not available; run the mean field first')
        self.e_corr, self.t2 = kernel(self, mo_energy, mo_coeff, eris, with_t2)
        log.note('Spinor MP2 correlation energy = %.15g', self.e_corr)
        return self.e_corr, self.t2

    def run(self, *args, **kwargs):
        self.kernel(*args, **kwargs)
        return self


MP2 = SpinorMP2


if __name__ == '__main__':
    from pyscf import gto, scf
    from socutils.scf import spinor_hf
    mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)
    mf = spinor_hf.SpinorSCF(mol).run()
    pt = SpinorMP2(mf).run()
    ref = scf.RHF(mol).run()
    from pyscf import mp
    print('spinor MP2 :', pt.e_corr)
    print('pyscf  MP2 :', mp.MP2(ref).run().e_corr)

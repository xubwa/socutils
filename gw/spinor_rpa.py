#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Relativistic (2-/4-component spinor) direct RPA (dRPA) correlation and total
energy.

This is the spinor analogue of PySCF's restricted ``gw.rpa``.  It evaluates the
adiabatic-connection / fluctuation-dissipation (ACFDT) RPA correlation energy on
the imaginary frequency axis,

    E_c^RPA = 1/(2*pi) * \\int_0^\\infty d(omega) Tr[ ln(1 - Pi(i*omega)) + Pi(i*omega) ]

where Pi is the (bare-Coulomb-dressed) RPA polarizability in the auxiliary
basis -- exactly the matrix built by
:func:`socutils.gw.spinor_gw_ac.get_rho_response`, which already carries the
spinor-correct prefactor of 2 (no spin sum) and the complex conjugates.  The
total RPA energy is

    E_tot^RPA = E_HF[reference density] + E_c^RPA ,

with ``E_HF`` the exact-exchange (Hartree-Fock) energy evaluated on the
reference orbitals; for a :class:`SpinorSCF` (Hartree-Fock) reference this is
just ``mf.e_tot``.

In the non-relativistic limit a SpinorSCF reference reproduces the restricted
RHF correlation energy of ``pyscf.gw.rpa``.

Method:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory Comput. 17, 727 (2021)
    X. Ren et al., New J. Phys. 14, 053020 (2012)
'''

import numpy
import numpy as np

from pyscf import lib
from pyscf import df
from pyscf.lib import logger
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__

from socutils.gw.spinor_gw_ac import (
    get_rho_response, _get_scaled_legendre_roots, _blas_single_thread,
    _mo_energy_without_core, _mo_without_core,
)

einsum = lib.einsum


def kernel(rpa, mo_energy, mo_coeff, Lov=None, nw=40, verbose=logger.NOTE):
    '''
    Spinor dRPA correlation energy.

    Returns:
        e_corr : float
    '''
    nocc = rpa.nocc
    if Lov is None:
        Lov = rpa.ao2mo(mo_coeff)
    naux = Lov.shape[0]

    freqs, wts = _get_scaled_legendre_roots(nw)

    def _ec_w(w):
        # Pi(i*omega) is Hermitian and negative semidefinite, so its
        # eigenvalues are real and <= 0; ln det(1-Pi)+Tr Pi = sum_k
        # [ln(1-lam_k) + lam_k].
        Pi = get_rho_response(freqs[w], mo_energy, Lov)
        lam = np.linalg.eigvalsh(Pi)
        return wts[w] / (2.0 * np.pi) * np.sum(np.log(1.0 - lam) + lam)

    nthreads = rpa.nthreads if rpa.nthreads else lib.num_threads()
    nthreads = min(nthreads, nw)
    if nthreads > 1:
        with _blas_single_thread():
            with lib.ThreadPoolExecutor(max_workers=nthreads) as ex:
                e_corr = sum(ex.map(_ec_w, range(nw)))
    else:
        e_corr = sum(_ec_w(w) for w in range(nw))

    return float(e_corr)


class SpinorRPA(lib.StreamObject):
    '''Relativistic spinor direct RPA (dRPA) correlation/total energy.

    Args:
        mf : a converged spinor mean-field object
            (:class:`socutils.scf.spinor_hf.SpinorSCF` or a subclass).
        frozen : int or list of int
            Frozen spinor orbitals, same convention as PySCF MP2 (int = frozen
            core, list = freeze those indices including virtuals).

    Attributes:
        nthreads : int
            Worker threads for the imaginary-frequency loop.

    Saved results:
        e_corr : RPA correlation energy
        e_hf   : exact-exchange (HF) energy on the reference
        e_tot  : e_hf + e_corr
    '''

    _keys = {
        'mol', 'frozen', 'with_df', 'nthreads',
        'e_corr', 'e_hf', 'e_tot', 'mo_energy', 'mo_coeff', 'mo_occ',
    }

    def __init__(self, mf, frozen=None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.frozen = frozen
        self.nthreads = lib.num_threads()

        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self._nocc = None
        self._nmo = None
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.e_corr = None
        self.e_hf = None
        self.e_tot = None

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('RPA (spinor) nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not None:
            log.info('frozen = %s', self.frozen)
        logger.info(self, 'nthreads (frequency loop) = %s', self.nthreads)
        return self

    @property
    def nocc(self):
        return self.get_nocc()

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    def get_frozen_mask(self):
        '''Boolean mask over the full spinor MO space; True = active orbital.
        Follows PySCF's MP2 ``frozen`` convention.'''
        return get_frozen_mask(self)

    get_nocc = get_nocc
    get_nmo = get_nmo

    def get_e_hf(self, mo_coeff=None):
        '''Exact-exchange (Hartree-Fock) energy evaluated on the reference
        density.  Equals mf.e_tot for a Hartree-Fock (SpinorSCF) reference.'''
        mf = self._scf
        dm = mf.make_rdm1()
        h1e = mf.get_hcore()
        vj, vk = mf.get_jk(self.mol, dm)
        e_hf = numpy.einsum('ij,ji->', h1e + 0.5 * (vj - vk), dm)
        return e_hf.real + mf.energy_nuc()

    def kernel(self, mo_energy=None, mo_coeff=None, Lov=None, nw=40):
        '''
        Input:
            nw : number of imaginary-frequency grid points.
        Output:
            e_corr : RPA correlation energy.
        '''
        if mo_coeff is None:
            mo_coeff = _mo_without_core(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = _mo_energy_without_core(self, self._scf.mo_energy)

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()

        self.e_hf = self.get_e_hf()
        self.e_corr = kernel(self, mo_energy, mo_coeff, Lov=Lov, nw=nw,
                             verbose=self.verbose)
        self.e_tot = self.e_hf + self.e_corr

        logger.note(self, 'RPA e_tot = %.12g  e_hf = %.12g  e_corr = %.12g',
                    self.e_tot, self.e_hf, self.e_corr)
        logger.timer(self, 'RPA', *cput0)
        return self.e_corr

    def ao2mo(self, mo_coeff=None):
        '''Build the complex spinor occ-vir DF tensor Lov[P,i,a].

        Both orbital indices are restricted (occ rows, vir columns), so the
        result is (naux, nocc, nvir) -- the only block dRPA needs.
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        mol = self.mol
        nao = mol.nao_nr()
        nocc = self.nocc
        nmo = mo_coeff.shape[1]
        nvir = nmo - nocc
        naux = self.with_df.get_naoaux()

        c2 = numpy.vstack(mol.sph2spinor_coeff())          # (2*nao, n2c)
        Csph = c2 @ mo_coeff
        Cao = numpy.asarray(Csph[:nao, :nocc])             # alpha, occ
        Cbo = numpy.asarray(Csph[nao:, :nocc])             # beta,  occ
        Cav = numpy.asarray(Csph[:nao, nocc:])             # alpha, vir
        Cbv = numpy.asarray(Csph[nao:, nocc:])             # beta,  vir

        mem_incore = naux * nocc * nvir * 16 / 1e6
        if not (mem_incore + lib.current_memory()[0] < 0.99 * self.max_memory
                or mol.incore_anyway):
            logger.warn(self, 'Memory may not be enough for incore Lov!')

        Lov = numpy.empty((naux, nocc, nvir), dtype=numpy.complex128)
        p1 = 0
        for eri1 in self.with_df.loop():
            eri3c = lib.unpack_tril(numpy.asarray(eri1))   # (blk, nao, nao)
            b = eri3c.shape[0]
            tmp = einsum('Pmn,mi->Pin', eri3c, Cao.conj())
            blk = einsum('Pin,na->Pia', tmp, Cav)
            tmp = einsum('Pmn,mi->Pin', eri3c, Cbo.conj())
            blk += einsum('Pin,na->Pia', tmp, Cbv)
            Lov[p1:p1 + b] = blk
            p1 += b
        assert p1 == naux
        return Lov


RPA = dRPA = SpinorRPA


if __name__ == '__main__':
    from pyscf import gto
    from socutils.scf import spinor_hf

    mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=4)
    mf = spinor_hf.SpinorSCF(mol).density_fit()
    mf.kernel()

    myrpa = SpinorRPA(mf)
    myrpa.kernel()
    print('spinor RPA  e_corr = %.10f' % myrpa.e_corr)
    print('spinor RPA  e_tot  = %.10f' % myrpa.e_tot)

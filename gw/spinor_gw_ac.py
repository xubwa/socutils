#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Relativistic (2-/4-component spinor) G0W0 approximation with analytic
continuation.

This module is the spinor analogue of PySCF's restricted ``gw.gw_ac``.  It
takes a spinor mean-field reference (:class:`socutils.scf.spinor_hf.SpinorSCF`,
or any of its X2C/X2CAMF subclasses) and computes the diagonal G0W0
quasiparticle energies on the imaginary frequency axis, then analytically
continues to the real axis.

Differences with respect to the non-relativistic restricted code:

* The molecular orbitals are complex two-component spinors, so the density
  fitting three-index tensor ``Lpq`` is complex and Hermitian in its orbital
  indices, ``Lpq[P,p,q] = conj(Lpq[P,q,p])``.  Every contraction that assumed
  real integrals in ``gw_ac`` is replaced by its complex-conjugate-aware
  counterpart.
* There is no spin degeneracy factor.  Each occupied spinor contributes a
  single fermion, so the RPA density response carries a prefactor of 2 (from
  combining the +i\\omega and -i\\omega poles) rather than the prefactor of 4
  (2 * 2 with the spin sum) used in the restricted code.

In the non-relativistic limit (a :class:`SpinorSCF` reference built without
spin-orbit coupling), each spatial orbital becomes a Kramers doublet and the
spinor-GW quasiparticle energies reproduce the restricted RHF-GW values, every
level doubly degenerate.

Method:
    See T. Zhu and G.K.-L. Chan, arxiv:2007.03148 (2020) for the AC-GW scheme
    that this implementation follows.
'''

from functools import reduce
import numpy
import numpy as np
from scipy.optimize import newton, least_squares

from pyscf import lib
from pyscf import df
from pyscf.lib import logger
from pyscf import __config__

einsum = lib.einsum


def kernel(gw, mo_energy, mo_coeff, Lpq=None, orbs=None,
           nw=None, vhf_df=False, verbose=logger.NOTE):
    '''
    Spinor GW quasiparticle orbital energies

    Returns:
        A list : converged, mo_energy, mo_coeff
    '''
    mf = gw._scf
    if gw.frozen is None:
        frozen = 0
    else:
        frozen = gw.frozen
    assert isinstance(frozen, int)
    assert frozen < gw.nocc

    if Lpq is None:
        Lpq = gw.ao2mo(mo_coeff)
    if orbs is None:
        orbs = range(gw.nmo)
    else:
        orbs = [x - frozen for x in orbs]
        if orbs[0] < 0:
            logger.warn(gw, 'GW orbs must be larger than frozen core!')
            raise RuntimeError

    nocc = gw.nocc
    nmo = gw.nmo
    naux = Lpq.shape[0]

    # v_xc (mean-field effective potential minus the Hartree term), in the MO
    # basis.  For a pure HF reference this equals the exact exchange and exactly
    # cancels vk below; for a DFT reference it is the xc potential.
    dm = mf.make_rdm1()
    v_mf = mf.get_veff(gw.mol, dm) - mf.get_j(gw.mol, dm)
    v_mf = reduce(numpy.dot, (mo_coeff.conj().T, v_mf, mo_coeff))
    v_mf = v_mf[frozen:, frozen:]

    # v_hf: exact (bare) exchange self-energy Sigma_x = -K
    if vhf_df and frozen == 0:
        vk = -einsum('Lni,Lim->nm', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
    else:
        vk = -mf.get_k(gw.mol, dm)
        vk = reduce(numpy.dot, (mo_coeff.conj().T, vk, mo_coeff))
        vk = vk[frozen:, frozen:]

    # Grids for integration on the imaginary axis
    freqs, wts = _get_scaled_legendre_roots(nw)

    # Self-energy (diagonal) on the imaginary axis i*[0, iw_cutoff]
    sigmaI, omega = get_sigma_diag(gw, orbs, Lpq, freqs, wts, iw_cutoff=5.)

    # Analytic continuation
    if gw.ac == 'twopole':
        coeff = AC_twopole_diag(sigmaI, omega, orbs, nocc)
    elif gw.ac == 'pade':
        coeff, omega_fit = AC_pade_thiele_diag(sigmaI, omega)

    conv = True
    mf_mo_energy = mo_energy.copy()
    ef = (mo_energy[nocc - 1] + mo_energy[nocc]) / 2.
    mo_energy = np.zeros_like(gw._scf.mo_energy)
    for p in orbs:
        if gw.linearized:
            # linearized G0W0
            de = 1e-6
            ep = mf_mo_energy[p]
            if gw.ac == 'twopole':
                sigmaR = two_pole(ep - ef, coeff[:, p - orbs[0]]).real
                dsigma = two_pole(ep - ef + de, coeff[:, p - orbs[0]]).real - sigmaR.real
            elif gw.ac == 'pade':
                sigmaR = pade_thiele(ep - ef, omega_fit[p - orbs[0]], coeff[:, p - orbs[0]]).real
                dsigma = pade_thiele(ep - ef + de, omega_fit[p - orbs[0]], coeff[:, p - orbs[0]]).real - sigmaR.real
            zn = 1.0 / (1.0 - dsigma / de)
            e = ep + zn * (sigmaR.real + vk[p, p].real - v_mf[p, p].real)
            mo_energy[p + frozen] = e
        else:
            # self-consistently solve the QP equation
            def quasiparticle(omega):
                if gw.ac == 'twopole':
                    sigmaR = two_pole(omega - ef, coeff[:, p - orbs[0]]).real
                elif gw.ac == 'pade':
                    sigmaR = pade_thiele(omega - ef, omega_fit[p - orbs[0]], coeff[:, p - orbs[0]]).real
                return omega - mf_mo_energy[p] - (sigmaR.real + vk[p, p].real - v_mf[p, p].real)
            try:
                e = newton(quasiparticle, mf_mo_energy[p], tol=1e-6, maxiter=100)
                mo_energy[p + frozen] = e
            except RuntimeError:
                conv = False

    if gw.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmo)
        logger.debug(gw, '  GW mo_energy =\n%s', mo_energy)
        numpy.set_printoptions(threshold=1000)

    return conv, mo_energy, mo_coeff


def get_rho_response(omega, mo_energy, Lpq):
    '''
    Compute the density response function in the auxiliary basis at frequency iw.

    For complex spinor orbitals the polarizability is

        Pi_PQ(iw) = 2 * sum_ia L^P_ia conj(L^Q_ia) (e_i - e_a)/(w^2 + (e_a-e_i)^2)

    The prefactor 2 (rather than 4 in the restricted code) reflects the absence
    of a separate spin sum: each spinor is a single fermion.
    '''
    naux, nocc, nvir = Lpq.shape
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    eia = eia / (omega**2 + eia * eia)
    Pia = einsum('Pia,ia->Pia', Lpq, eia)
    # No spin degeneracy factor; conjugate the right index for complex MOs.
    Pi = 2. * einsum('Pia,Qia->PQ', Pia, Lpq.conj())
    return Pi


def get_sigma_diag(gw, orbs, Lpq, freqs, wts, iw_cutoff=None):
    '''
    Compute the (diagonal) GW correlation self-energy in the MO basis on the
    imaginary axis.

    Because ``Lpq`` is Hermitian in its orbital indices, the contraction
    ``Wmn = einsum('Qnm,Qmn->mn', Qnm, Lpq[:,:,orbs])`` automatically supplies
    the complex conjugate required by the diagonal self-energy
    Sigma_nn = sum_m sum_PQ L^P_nm W_PQ conj(L^Q_nm) g0_m.
    '''
    mo_energy = _mo_energy_without_core(gw, gw._scf.mo_energy)
    nocc = gw.nocc
    nw = len(freqs)
    naux = Lpq.shape[0]
    norbs = len(orbs)

    # TODO: Treatment of degeneracy
    if (mo_energy[nocc] - mo_energy[nocc - 1]) < 1e-3:
        logger.warn(gw, 'GW not well-defined for degeneracy!')
    ef = (mo_energy[nocc - 1] + mo_energy[nocc]) / 2.

    # Integration on numerical grids
    if iw_cutoff is not None:
        nw_sigma = sum(iw < iw_cutoff for iw in freqs) + 1
    else:
        nw_sigma = nw + 1

    # Compute occ for -iw and vir for iw separately
    # to avoid branch cuts in the analytic continuation
    omega_occ = np.zeros((nw_sigma), dtype=np.complex128)
    omega_vir = np.zeros((nw_sigma), dtype=np.complex128)
    omega_occ[1:] = -1j * freqs[:(nw_sigma - 1)]
    omega_vir[1:] = 1j * freqs[:(nw_sigma - 1)]
    orbs_occ = [i for i in orbs if i < nocc]
    norbs_occ = len(orbs_occ)

    emo_occ = omega_occ[None, :] + ef - mo_energy[:, None]
    emo_vir = omega_vir[None, :] + ef - mo_energy[:, None]

    sigma = np.zeros((norbs, nw_sigma), dtype=np.complex128)
    omega = np.zeros((norbs, nw_sigma), dtype=np.complex128)
    for p in range(norbs):
        orbp = orbs[p]
        if orbp < nocc:
            omega[p] = omega_occ.copy()
        else:
            omega[p] = omega_vir.copy()

    for w in range(nw):
        Pi = get_rho_response(freqs[w], mo_energy, Lpq[:, :nocc, nocc:])
        Pi_inv = np.linalg.inv(np.eye(naux) - Pi) - np.eye(naux)
        g0_occ = wts[w] * emo_occ / (emo_occ**2 + freqs[w]**2)
        g0_vir = wts[w] * emo_vir / (emo_vir**2 + freqs[w]**2)
        Qnm = einsum('Pnm,PQ->Qnm', Lpq[:, orbs, :], Pi_inv)
        Wmn = einsum('Qnm,Qmn->mn', Qnm, Lpq[:, :, orbs])
        sigma[:norbs_occ] += -einsum('mn,mw->nw', Wmn[:, :norbs_occ], g0_occ) / np.pi
        sigma[norbs_occ:] += -einsum('mn,mw->nw', Wmn[:, norbs_occ:], g0_vir) / np.pi

    return sigma, omega


def _get_scaled_legendre_roots(nw):
    """
    Scale nw Legendre roots, which lie in the interval [-1, 1], so that they lie
    in [0, inf).
    Ref: www.cond-mat.de/events/correl19/manuscripts/ren.pdf

    Returns:
        freqs : 1D ndarray
        wts : 1D ndarray
    """
    freqs, wts = np.polynomial.legendre.leggauss(nw)
    x0 = 0.5
    freqs_new = x0 * (1. + freqs) / (1. - freqs)
    wts = wts * 2. * x0 / (1. - freqs)**2
    return freqs_new, wts


def _get_clenshaw_curtis_roots(nw):
    """
    Clenshaw-Curtis quadrature on [0,inf).
    Ref: J. Chem. Phys. 132, 234114 (2010)

    Returns:
        freqs : 1D ndarray
        wts : 1D ndarray
    """
    freqs = np.zeros(nw)
    wts = np.zeros(nw)
    a = 0.2
    for w in range(nw):
        t = (w + 1.0) / nw * np.pi / 2.
        freqs[w] = a / np.tan(t)
        if w != nw - 1:
            wts[w] = a * np.pi / 2. / nw / (np.sin(t)**2)
        else:
            wts[w] = a * np.pi / 4. / nw / (np.sin(t)**2)
    return freqs[::-1], wts[::-1]


def two_pole_fit(coeff, omega, sigma):
    cf = coeff[:5] + 1j * coeff[5:]
    f = cf[0] + cf[1] / (omega + cf[3]) + cf[2] / (omega + cf[4]) - sigma
    f[0] = f[0] / 0.01
    return np.array([f.real, f.imag]).reshape(-1)


def two_pole(freqs, coeff):
    cf = coeff[:5] + 1j * coeff[5:]
    return cf[0] + cf[1] / (freqs + cf[3]) + cf[2] / (freqs + cf[4])


def AC_twopole_diag(sigma, omega, orbs, nocc):
    """
    Analytic continuation to the real axis using a two-pole model.

    Returns:
        coeff: 2D array (ncoeff, norbs)
    """
    norbs, nw = sigma.shape
    coeff = np.zeros((10, norbs))
    for p in range(norbs):
        if orbs[p] < nocc:
            x0 = np.array([0, 1, 1, 1, -1, 0, 0, 0, -1.0, -0.5])
        else:
            x0 = np.array([0, 1, 1, 1, -1, 0, 0, 0, 1.0, 0.5])
        xopt = least_squares(two_pole_fit, x0, jac='3-point', method='trf', xtol=1e-10,
                             gtol=1e-10, max_nfev=1000, verbose=0, args=(omega[p], sigma[p]))
        if xopt.success is False:
            print('WARN: 2P-Fit Orb %d not converged, cost function %e' % (p, xopt.cost))
        coeff[:, p] = xopt.x.copy()
    return coeff


def thiele(fn, zn):
    nfit = len(zn)
    g = np.zeros((nfit, nfit), dtype=np.complex128)
    g[:, 0] = fn.copy()
    for i in range(1, nfit):
        g[i:, i] = (g[i - 1, i - 1] - g[i:, i - 1]) / ((zn[i:] - zn[i - 1]) * g[i:, i - 1])
    a = g.diagonal()
    return a


def pade_thiele(freqs, zn, coeff):
    nfit = len(coeff)
    X = coeff[-1] * (freqs - zn[-2])
    for i in range(nfit - 1):
        idx = nfit - i - 1
        X = coeff[idx] * (freqs - zn[idx - 1]) / (1. + X)
    X = coeff[0] / (1. + X)
    return X


def AC_pade_thiele_diag(sigma, omega):
    """
    Analytic continuation to the real axis using a Pade approximation from
    Thiele's reciprocal difference method.
    Reference: J. Low Temp. Phys. 29, 179 (1977)

    Returns:
        coeff: 2D array (ncoeff, norbs)
        omega: 2D array (norbs, npade)
    """
    idx = range(1, 40, 6)
    sigma1 = sigma[:, idx].copy()
    sigma2 = sigma[:, (idx[-1] + 4)::4].copy()
    sigma = np.hstack((sigma1, sigma2))
    omega1 = omega[:, idx].copy()
    omega2 = omega[:, (idx[-1] + 4)::4].copy()
    omega = np.hstack((omega1, omega2))
    norbs, nw = sigma.shape
    npade = nw // 2
    coeff = np.zeros((npade * 2, norbs), dtype=np.complex128)
    for p in range(norbs):
        coeff[:, p] = thiele(sigma[p, :npade * 2], omega[p, :npade * 2])
    return coeff, omega[:, :npade * 2]


def _mo_energy_without_core(gw, mo_energy):
    frozen = 0 if gw.frozen is None else gw.frozen
    return mo_energy[frozen:]


def _mo_without_core(gw, mo):
    frozen = 0 if gw.frozen is None else gw.frozen
    return mo[:, frozen:]


class SpinorGWAC(lib.StreamObject):
    '''Relativistic spinor G0W0 with analytic continuation.

    Args:
        mf : a converged spinor mean-field object
            (:class:`socutils.scf.spinor_hf.SpinorSCF` or a subclass).
        frozen : int
            Number of frozen core *spinors* (frozen-core only).

    Saved results:
        mo_energy : 1D array
            The GW quasiparticle energies (only ``orbs`` are updated; the
            others are left at zero).
    '''

    linearized = getattr(__config__, 'gw_gw_GW_linearized', False)
    # Analytic continuation: 'pade' or 'twopole'
    ac = getattr(__config__, 'gw_gw_GW_ac', 'pade')

    _keys = {
        'linearized', 'ac', 'with_df', 'mol', 'frozen',
        'mo_energy', 'mo_coeff', 'mo_occ', 'sigma',
    }

    def __init__(self, mf, frozen=None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.frozen = frozen

        # DF-GW must use density fitting integrals
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

##################################################
# don't modify the following attributes, they are not input options
        self._nocc = None
        self._nmo = None
        # self.mo_energy: GW quasiparticle energy, not scf mo_energy
        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.sigma = None

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW (spinor) nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not None:
            log.info('frozen = %s', self.frozen)
        logger.info(self, 'use perturbative linearized QP eqn = %s', self.linearized)
        logger.info(self, 'analytic continuation method = %s', self.ac)
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

    def get_nocc(self):
        if self._nocc is not None:
            return self._nocc
        frozen = 0 if self.frozen is None else self.frozen
        nocc = int(numpy.sum(self._scf.mo_occ > 0))
        return nocc - frozen

    def get_nmo(self):
        if self._nmo is not None:
            return self._nmo
        frozen = 0 if self.frozen is None else self.frozen
        return len(self._scf.mo_energy) - frozen

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, orbs=None,
               nw=100, vhf_df=False):
        """
        Input:
            orbs : list of orbital indices (in the full, non-frozen-shifted
                   spinor space) to correct.
            nw : number of imaginary-frequency grid points.
            vhf_df : whether to use density fitting for the HF exchange.
        Output:
            mo_energy : GW quasiparticle energies.
        """
        if mo_coeff is None:
            mo_coeff = _mo_without_core(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = _mo_energy_without_core(self, self._scf.mo_energy)

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        self.converged, self.mo_energy, self.mo_coeff = \
            kernel(self, mo_energy, mo_coeff,
                   Lpq=Lpq, orbs=orbs, nw=nw, vhf_df=vhf_df, verbose=self.verbose)

        logger.warn(self, 'GW QP energies may not be sorted from min to max')
        logger.timer(self, 'GW', *cput0)
        return self.mo_energy

    def ao2mo(self, mo_coeff=None):
        '''Build the complex spinor DF tensor Lpq[P,p,q] in the MO basis.

        The DF three-index integrals (P|mu nu) are real and live in the real
        spherical AO basis.  The spinor MOs are projected to the spherical GHF
        basis (alpha/beta blocks of dimension nao) and contracted spin-block by
        spin-block:

            Lpq[P,p,q] = sum_{mu nu} (P|mu nu)
                         * (Ca*[mu,p] Ca[nu,q] + Cb*[mu,p] Cb[nu,q])
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        mol = self.mol
        nao = mol.nao_nr()
        nmo = mo_coeff.shape[1]
        naux = self.with_df.get_naoaux()

        c2 = numpy.vstack(mol.sph2spinor_coeff())   # (2*nao, n2c)
        Csph = c2 @ mo_coeff                         # (2*nao, nmo)
        Ca = numpy.asarray(Csph[:nao])
        Cb = numpy.asarray(Csph[nao:])

        mem_incore = naux * nmo**2 * 16 / 1e6
        mem_now = lib.current_memory()[0]
        if not ((mem_incore + mem_now < 0.99 * self.max_memory) or mol.incore_anyway):
            logger.warn(self, 'Memory may not be enough for incore Lpq!')

        Lpq = numpy.empty((naux, nmo, nmo), dtype=numpy.complex128)
        p1 = 0
        for eri1 in self.with_df.loop():
            eri3c = lib.unpack_tril(numpy.asarray(eri1))   # (blk, nao, nao)
            b = eri3c.shape[0]
            tmp = einsum('Pmn,mp->Ppn', eri3c, Ca.conj())
            blk = einsum('Ppn,nq->Ppq', tmp, Ca)
            tmp = einsum('Pmn,mp->Ppn', eri3c, Cb.conj())
            blk += einsum('Ppn,nq->Ppq', tmp, Cb)
            Lpq[p1:p1 + b] = blk
            p1 += b
        assert p1 == naux
        return Lpq


GWAC = SpinorGWAC


if __name__ == '__main__':
    from pyscf import gto, scf
    from socutils.scf import spinor_hf

    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = 'H 0 0 0; H 0 0 0.74'
    mol.basis = 'def2-svp'
    mol.build()

    mf = spinor_hf.SpinorSCF(mol).density_fit()
    mf.kernel()

    gw = SpinorGWAC(mf)
    gw.ac = 'pade'
    gw.linearized = False
    nocc = mol.nelectron
    gw.kernel(orbs=range(nocc - 2, nocc + 2))
    print('spinor GW QP energies:', gw.mo_energy[nocc - 2:nocc + 2])

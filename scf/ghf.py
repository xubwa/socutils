#!/usr/bin/env python
'''
Generalized Hartree-Fock (GHF) driver with an X2C spin-orbit Hamiltonian.

This is the spin-orbital (GHF) analogue of :mod:`socutils.scf.spinor_hf`.  The
spin-orbit physics is attached through ``with_x2c`` with the ``.x2camf()`` /
``.x2cmp()`` shortcuts, exactly as ``spinor_hf.SCF(mol).x2camf()`` does for the
spinor representation -- mirroring PySCF's ``mf.x2c()``::

    from socutils.scf import ghf
    mf = ghf.GHF(mol).x2camf()
    mf.kernel()

The spinor and GHF styles describe the same Hamiltonian and agree to numerical
precision; the GHF style stays close to PySCF's native GHF toolchain.
'''

import numpy
from pyscf.scf import ghf, hf
from pyscf.lib import logger


def _is_kramers_even(dms, nao, tol=1e-10):
    '''Every dm Hermitian and time-reversal even: D_bb = D_aa^*, D_ba = -D_ab^*
    (a Kramers closed-shell density, and its time-even response).'''
    aa, ab = dms[:, :nao, :nao], dms[:, :nao, nao:]
    ba, bb = dms[:, nao:, :nao], dms[:, nao:, nao:]
    scale = max(abs(dms).max(), 1e-300)
    return (abs(bb - aa.conj()).max() < tol * scale
            and abs(ba + ab.conj()).max() < tol * scale)


def get_jk_kramers(mf, mol, dms, with_j=True, with_k=True, omega=None):
    '''J and K of Hermitian, time-reversal-even GHF densities from four real
    scalar-AO builds: with D_aa = R + iA (R symmetric, A anti-symmetric) and
    D_ab = P + iQ (both anti-symmetric for such a density),
        J_aa = J_bb = J[2R],  K_aa = K[R] + i K[A] = K_bb^*,
        K_ab = K[P] + i K[Q],  K_ba = K_ab^H,
    (pyscf.scf.ghf.get_jk builds K of the three complex blocks with hermi=0).
    Without spin-orbit coupling A = P = Q = 0: one symmetric J/K build, as RHF.'''
    nao = mol.nao_nr()
    n_dm = dms.shape[0]
    R = numpy.ascontiguousarray(dms[:, :nao, :nao].real)
    A = numpy.ascontiguousarray(dms[:, :nao, :nao].imag)
    P = numpy.ascontiguousarray(dms[:, :nao, nao:].real)
    Q = numpy.ascontiguousarray(dms[:, :nao, nao:].imag)
    # hf.RHF.get_jk as pyscf.scf.ghf: in-core integrals when they fit
    jr, kr = hf.RHF.get_jk(mf, mol, R, 1, with_j, with_k, omega)
    vj = vk = None
    if with_j:
        vj = numpy.zeros((n_dm, 2 * nao, 2 * nao), dtype=dms.dtype)
        vj[:, :nao, :nao] = vj[:, nao:, nao:] = 2 * numpy.asarray(jr).reshape(n_dm, nao, nao)
    if with_k:
        kr = numpy.asarray(kr).reshape(n_dm, nao, nao)
        anti = [M for M in (A, P, Q)]
        live = [i for i, M in enumerate(anti) if abs(M).max() > 0]
        ka = {}
        if live:
            stack = numpy.concatenate([anti[i] for i in live])
            ks = hf.RHF.get_jk(mf, mol, stack, 2, False, True, omega)[1]
            ks = numpy.asarray(ks).reshape(len(live), n_dm, nao, nao)
            ka = {i: ks[m] for m, i in enumerate(live)}
        zero = numpy.zeros((n_dm, nao, nao))
        kA, kP, kQ = (ka.get(i, zero) for i in range(3))
        if numpy.iscomplexobj(dms):
            kaa, kab = kr + 1j * kA, kP + 1j * kQ
        else:
            kaa, kab = kr, kP
        vk = numpy.empty((n_dm, 2 * nao, 2 * nao), dtype=kaa.dtype)
        vk[:, :nao, :nao] = kaa
        vk[:, nao:, nao:] = kaa.conj()
        vk[:, :nao, nao:] = kab
        vk[:, nao:, :nao] = kab.conj().transpose(0, 2, 1)
    return vj, vk


class GHF(ghf.GHF):
    '''GHF with an X2C spin-orbit Hamiltonian attached through ``with_x2c``.'''

    _keys = {'with_x2c'}

    def __init__(self, mol):
        ghf.GHF.__init__(self, mol)
        self.with_x2c = None

    def cholesky(self, tau=1e-4, sigma=1e-2, method='threeloop',
                 cderi=None, cderi_to_save=None, with_df=None, **kwargs):
        '''Cholesky-decomposed ERIs -- the CD analogue of ``mf.density_fit()``.'''
        from socutils.cd.cd import cholesky
        return cholesky(self, tau=tau, sigma=sigma, method=method,
                        cderi=cderi, cderi_to_save=cderi_to_save,
                        with_df=with_df, **kwargs)

    def _attach_x2cmp(self, flavor, with_gaunt, with_breit, with_pcc, with_aoc):
        from socutils.somf.x2cmp import SpinOrbitalX2CMPHelper
        self.with_x2c = SpinOrbitalX2CMPHelper(
            self.mol, x2cmp=flavor, with_gaunt=with_gaunt, with_breit=with_breit,
            with_pcc=with_pcc, with_aoc=with_aoc)
        self._keys = self._keys.union(['with_x2c'])
        terms = ['Dirac-Coulomb']
        if with_gaunt: terms.append('Gaunt')
        if with_breit: terms.append('Breit')
        if with_pcc: terms.append('PCC')
        logger.info(self, 'X2C SOC Hamiltonian: flavor=%s, terms = %s',
                    flavor, ' + '.join(terms))
        return self

    def x2camf(self, flavor='x2camf', with_gaunt=True, with_breit=True,
               with_pcc=False, with_aoc=False):
        '''Attach an X2CAMF (atomic-mean-field) spin-orbit Hamiltonian to this
        GHF object, in place, and return it -- the GHF analogue of PySCF's
        ``mf.x2c()``.  The SOC enters through ``self.with_x2c`` and is applied
        in :meth:`get_hcore`.

        >>> mf = GHF(mol).x2camf()               # Gaunt + Breit
        >>> mf.kernel()
        '''
        return self._attach_x2cmp(flavor, with_gaunt, with_breit, with_pcc, with_aoc)

    def x2cmp(self, flavor='x2cmp', with_gaunt=True, with_breit=True,
              with_pcc=False, with_aoc=False):
        '''Attach an X2C molecular-mean-field (molecular picture-change) spin-
        orbit Hamiltonian to this GHF object, in place, and return it.  Same
        usage as :meth:`x2camf`, with the molecular-mean-field flavor.

        >>> mf = GHF(mol).x2cmp()
        >>> mf.kernel()
        '''
        return self._attach_x2cmp(flavor, with_gaunt, with_breit, with_pcc, with_aoc)

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        if self.with_x2c is not None:
            return self.with_x2c.get_hcore(mol)
        return ghf.GHF.get_hcore(self, mol)

    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True, omega=None):
        '''pyscf's GHF J/K; Hermitian time-reversal-even densities (Kramers
        closed shell, its time-even response) through get_jk_kramers.'''
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        dm = numpy.asarray(dm)
        nao = mol.nao_nr()
        if (hermi == 1 and dm.shape[-1] == 2 * nao and getattr(self, 'with_df', None) is None
                and self.kramers_jk):
            dms = dm.reshape(-1, 2 * nao, 2 * nao)
            if _is_kramers_even(dms, nao):
                vj, vk = get_jk_kramers(self, mol, dms, with_j, with_k, omega)
                if vj is not None:
                    vj = vj.reshape(dm.shape)
                if vk is not None:
                    vk = vk.reshape(dm.shape)
                return vj, vk
        return ghf.GHF.get_jk(self, mol, dm, hermi, with_j, with_k, omega)

    # use get_jk_kramers for time-reversal-even densities
    kramers_jk = True


SCF = GHF

#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
X2CMMF: the two-component reference of a converged four-component Dirac-HF.

The converged 4c Fock matrix is block-diagonalized by the exact two-component
(X2C) decoupling,

    f_X2CMMF = R^+ L R,     C_2c = R^-1 C_L,

where X, R come from the positive-energy eigenvectors of the 4c Fock itself,
so the two-component orbitals are the positive-energy DHF orbitals, the
orbital energies are the positive-energy DHF eigenvalues and the reference
energy is the DHF energy.  Correlated methods then use f_X2CMMF with the
untransformed (nonrelativistic) two-electron integrals.

The resulting object is a converged spinor SCF that cannot be changed: its
Fock matrix and energy are defined by the DHF density only.  The X2CMMF
one-electron operator is the 4c mean field folded with a variational density,
not an atomic model potential (as in X2CMP), so moving the density away from
the DHF one (rerunning the SCF, changing occupations) has no meaning.
Accordingly kernel() and get_hcore() raise, and get_fock() / energy_elec()
accept only the reference density.  Orbital response on this reference
(stability, CPHF, gradients) is the response of the X2CMMF model, not of the
4c Hamiltonian.

Ref: X. Wang, C. Zhang, J. Liu, L. Cheng, arXiv:2504.19479.
'''

import numpy
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import dhf
from socutils.scf import spinor_hf
from socutils.scf import double_group_symm
from socutils.somf.x2c_grad import x2c1e_hfw0_block


def x2cmmf(mf):
    '''The X2CMMF two-component reference of a converged DHF object.

    DHF / RDHF give a Kramers-unrestricted / -restricted spinor SCF object;
    with double-group symmetry (SymAdaptedDHF / SymAdaptedRDHF) the object
    is symmetry adapted and the orbitals keep their irreps.
    '''
    assert isinstance(mf, dhf.DHF)
    if not mf.converged:
        logger.warn(mf, 'X2CMMF reference built from a DHF that is not converged')
    mol = mf.mol
    n2c = mol.nao_2c()
    log = logger.new_logger(mf)

    dm = mf.make_rdm1()
    vhf = mf.get_veff(mol, dm)
    fock4c = mf.get_fock(vhf=vhf, dm=dm)
    e_elec, e_coul = mf.energy_elec(dm, vhf=vhf)
    s4c = mf.get_ovlp()
    r, l = x2c1e_hfw0_block(fock4c[:n2c, :n2c], fock4c[n2c:, :n2c], fock4c[:n2c, n2c:],
                            fock4c[n2c:, n2c:], s4c[:n2c, :n2c], s4c[n2c:, n2c:])[4:6]
    fock2c = reduce(numpy.dot, (r.T.conj(), l, r))
    mo_coeff = numpy.linalg.solve(r, mf.mo_coeff[:n2c, n2c:])

    kramers = isinstance(mf, dhf.RDHF)
    if isinstance(mf, double_group_symm.DoubleGroupSymm):
        cls = SymAdaptedRHF if kramers else SymAdaptedUHF
        mf2c = cls(mol)
        mf2c._doublegroupname = mf._doublegroupname
        mf2c.irrep_nelec = dict(mf.irrep_nelec)
        orbsym = mf.get_orbsym(mf.mo_coeff)[n2c:]
        mo_coeff = lib.tag_array(mo_coeff, orbsym=orbsym)
    else:
        mf2c = (RHF if kramers else UHF)(mol)
    mf2c.verbose = mf.verbose
    mf2c.stdout = mf.stdout
    mf2c.with_gaunt = getattr(mf, 'with_gaunt', False)
    mf2c.with_breit = getattr(mf, 'with_breit', False)

    mf2c.fock_x2cmmf = fock2c
    mf2c._e_elec_4c = (e_elec, e_coul)
    mf2c.mo_coeff = mo_coeff
    mf2c.mo_occ = numpy.asarray(mf.mo_occ[n2c:])
    mf2c.mo_energy = reduce(numpy.dot, (mo_coeff.conj().T, fock2c, mo_coeff)).diagonal().real
    mf2c._dm_ref = mf2c.make_rdm1()
    mf2c.e_tot = e_elec + mf.energy_nuc()
    mf2c.converged = mf.converged
    log.info('X2CMMF reference: E = %.15g (DHF energy)', mf2c.e_tot)
    return mf2c


class _X2CMMF:
    '''Fixed X2CMMF reference; see the module docstring.

    Attributes:
        fock_x2cmmf : 2D array
            The X2C-transformed DHF Fock matrix in the spinor AO basis.
        with_gaunt, with_breit : bool
            The two-electron Hamiltonian of the parent DHF (information only).
    '''

    _keys = {'fock_x2cmmf', 'with_gaunt', 'with_breit'}
    # tolerance for a density matrix to count as the reference one
    _dm_tol = 1e-8

    fock_x2cmmf = None
    with_gaunt = False
    with_breit = False
    _e_elec_4c = None
    _dm_ref = None

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, 'X2CMMF reference from DHF (gaunt %s, breit %s)', self.with_gaunt, self.with_breit)
        return self

    def kernel(self, *args, **kwargs):
        raise RuntimeError('The X2CMMF reference is defined by the DHF solution and has no SCF of its own. '
                           'To change the reference, rerun the DHF calculation.')

    scf = kernel

    def get_hcore(self, mol=None):
        raise NotImplementedError('The X2CMMF reference has no one-electron Hamiltonian of its own; '
                                  'use get_fock() (the X2C-transformed DHF Fock matrix).')

    def _check_dm(self, dm):
        if dm is None:
            return
        dm = numpy.asarray(dm)
        if dm.shape != self._dm_ref.shape or abs(dm - self._dm_ref).max() > self._dm_tol:
            raise ValueError('The X2CMMF Fock matrix and energy are defined for the reference '
                             'density only.')

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                 diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
                 fock_last=None):
        self._check_dm(dm)
        return self.fock_x2cmmf

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        '''The electronic energy (and Coulomb part) of the parent DHF.'''
        self._check_dm(dm)
        e_elec, e_coul = self._e_elec_4c
        self.scf_summary['e1'] = e_elec - e_coul
        self.scf_summary['e2'] = e_coul
        return e_elec, e_coul


class UHF(_X2CMMF, spinor_hf.UHF):
    pass


class RHF(_X2CMMF, spinor_hf.RHF):
    pass


class SymAdaptedUHF(_X2CMMF, double_group_symm.SymAdaptedUHF):
    pass


class SymAdaptedRHF(_X2CMMF, double_group_symm.SymAdaptedRHF):
    pass


dhf.DHF.x2cmmf = x2cmmf

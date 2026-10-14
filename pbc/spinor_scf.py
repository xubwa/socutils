#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
# Gamma-point two-component (j-adapted spinor) SCF for a cell, with the
# Coulomb interactions from socutils.pbc.rsdf.RSDF.
#
# The molecular SpinorSCF machinery is reused unchanged: RSDF plays the role
# of the density-fitting object (scf.spinor_hf._DFJHF for J and the MO-based
# K, mcscf.zmc_ao2mo._CDERIS for the CASCI/CASSCF integral transformation),
# and the one-electron part is replaced by its periodic counterpart:
#     S, T      lattice sums at Gamma (cell.pbc_intor)
#     V_ne      RSDF.get_nuc(): Gaussian nuclei through the same kernel as J
#     E_nn      RSDF.energy_nuc()
#     relativity  X2CMP in its one-centre approximation (= X2CAMF), see
#               OneCenterX2CMPHelper: mf.x2camf() or mf.x2cmp('x2cmp_1c')
# All Coulomb terms use one convention (G = 0 removed, tin-foil), so the
# total energy of the neutral cell is well defined.
#
# Exchange finite-size term (exxdiv): the two-electron integrals are those of
# the G = 0-free kernel.  The G = 0 part of (pq|rs) is v delta_pq delta_rs, so
# for ANY N-electron wavefunction it contributes v N(N-1)/2, and with the
# neutralising background the constant -nu_M N / 2 (nu_M: Madelung constant of
# the cell).  exxdiv='ewald' (default, as in pyscf) adds this constant to
# energy_nuc(); it equals pyscf's ewald exchange correction for HF at
# convergence and leaves orbitals, CI vectors and correlation energies
# unchanged, so CASCI/CASSCF/CAHF on top see the same constant.  exxdiv=None
# gives the bare G = 0-free energy (error ~ 1/L for a molecule in a box).
#
# A fractional-occupation mean field is not an N-electron wavefunction: its
# exchange G = 0 term is -nu_M tr(DSDS)/2, not -nu_M N/2.  The SCF therefore
# uses pyscf's form, K + nu_M S D S in the Fock matrix and the energy (with
# the constant taken back out of energy_nuc); for an idempotent density this
# only shifts the occupied block and gives the same energy.  (O atom, 2p^4 at
# 2/3 in a box: with the constant the fractional HF energy missed the
# molecular one by 5e-2 Eh.)
#
# RSDF(lr='gspace') keeps the G = 0 part of its SR kernel, -c0 S_mn S_ls,
# out of the vectors (pbc/rsdf.py); it is handled like the exchange G = 0
# term: explicitly in the mean field (g0_terms), as the constant
# -c0 N (N - 1)/2 in energy_nuc for wavefunctions (wavefunction_constant).
#
# Usage:
#     r = RSDF(cell, auxbasis).build()
#     mf = spinor_scf.SpinorRSDF(cell, r).run()
#     mc = zcasci.CASCI(mf, ncas, nelecas).run()
#

import numpy
from pyscf import lib
from socutils.scf import spinor_hf


# periodic flavors: the one-centre approximation of X2CMP, keyed by the
# molecular flavor used for the isolated atoms
ONE_CENTER_FLAVORS = {'x2camf': 'x2camf',      # X2C-1e + atomic mean-field SOC
                      'x2cmp_1c': 'x2cmp'}     # atomic X2CMP (atomic Fock decoupled)


class OneCenterX2CMPHelper:
    '''X2CMP in the one-centre approximation for a cell (or any molecule).

    The relativistic correction of the one-electron Hamiltonian is a sum of
    atomic blocks,
        h = h_NR + sum_A [h_X2C(A) - h_NR(A)],
    where h_X2C(A) is the molecular socutils X2C Hamiltonian of the isolated
    atom A in its own basis (flavor 'x2camf': X2C-1e decoupling plus the
    atomic mean-field SOC, i.e. X2CAMF; 'x2cmp_1c': the atomic 4c Fock
    matrix decoupled), and h_NR(A) = T + V_A of the same atom.  All
    multi-centre relativistic terms (other nuclei inside the decoupling, the
    two-centre picture change) are dropped: that is what X2CAMF is relative to
    X2CMP, and what makes it applicable to a crystal.  h_NR of the cell (T and
    the periodic V_ne) is supplied by the caller.
    '''
    def __init__(self, mol, flavor='x2camf', with_gaunt=True, with_breit=True,
                 with_pcc=False, with_aoc=False):
        if flavor not in ONE_CENTER_FLAVORS:
            raise ValueError('periodic X2C: only the one-centre approximation of X2CMP is '
                             'available, flavor %s (got %r)' % (' or '.join(map(repr, ONE_CENTER_FLAVORS)), flavor))
        self.mol = mol
        self.flavor = flavor
        self.gaunt = with_gaunt
        self.breit = with_breit
        self.pcc = with_pcc
        self.aoc = with_aoc
        self.atomic = {}      # element -> h_X2C(A) - h_NR(A), spinor basis of A
        self._delta = None

    def _atom_correction(self, ia):
        from pyscf import gto
        from socutils.somf.x2cmp import SpinorX2CMPHelper
        mol = self.mol
        key = mol.atom_symbol(ia)
        if key not in self.atomic:
            atom = gto.Mole()
            atom.atom = [(key, (0., 0., 0.))]
            atom.unit = 'Bohr'
            atom.basis = {key: mol._basis[key]}
            atom.nucmod = {key: mol.nucmod[key]} if isinstance(mol.nucmod, dict) and key in mol.nucmod else {}
            atom.cart = mol.cart
            atom.verbose = 0
            atom.spin = gto.charge(key) % 2
            atom.build(False, False)
            x2c = SpinorX2CMPHelper(atom, x2cmp=ONE_CENTER_FLAVORS[self.flavor], with_gaunt=self.gaunt,
                                    with_breit=self.breit, with_pcc=self.pcc, with_aoc=self.aoc)
            h_x2c = x2c.get_hcore(atom)
            h_nr = atom.intor('int1e_kin_spinor') + atom.intor('int1e_nuc_spinor')
            self.atomic[key] = h_x2c - h_nr
        return self.atomic[key]

    def get_correction(self):
        '''sum_A [h_X2C(A) - h_NR(A)] in the spinor basis of mol (block diagonal by atom).'''
        if self._delta is None:
            mol = self.mol
            n2c = mol.nao_2c()
            d = numpy.zeros((n2c, n2c), dtype=complex)
            for ia, (_, _, p0, p1) in enumerate(mol.aoslice_2c_by_atom()):
                da = self._atom_correction(ia)
                assert da.shape == (p1 - p0, p1 - p0)
                d[p0:p1, p0:p1] = da
            self._delta = d
        return self._delta


def _sph_blockdiag(h):
    nao = h.shape[-1]
    out = numpy.zeros((2 * nao, 2 * nao), dtype=h.dtype)
    out[:nao, :nao] = out[nao:, nao:] = h
    return out


class _PeriodicSpinorSCF(spinor_hf._DFJHF):
    '''Mixin over a molecular spinor SCF class: periodic one-electron integrals
    and RSDF Coulomb interactions (with_df).'''

    exxdiv = 'ewald'
    _keys = {'exxdiv'}

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        vj, vk = self.get_jk(mol, dm, hermi=hermi)
        dvj, dvk = self.g0_terms(dm)
        return vj + dvj - (vk + dvk)

    def k_shift(self):
        '''Coefficient of S D S that the mean field adds to K: the exchange
        G = 0 term nu_M minus the SR G = 0 part kept out of the RSDF vectors
        (lr='gspace').'''
        return self.madelung() - getattr(self.with_df, 'c_g0', 0.)

    def g0_terms(self, dm):
        '''(dJ, dK) the mean field adds to mf.get_jk: J gets -c0 tr(DS) S (the
        G = 0 part of the SR kernel kept out of the vectors), K gets
        (nu_M - c0) S D S.  Zero for lr='fit' with exxdiv=None.'''
        c0 = getattr(self.with_df, 'c_g0', 0.)
        sk = self.k_shift()
        if not c0 and not sk:
            return 0, 0
        s = self.get_ovlp()
        dvj = -c0 * numpy.einsum('ij,ji', dm, s).real * s if c0 else 0
        dvk = sk * (s @ dm @ s) if sk else 0
        return dvj, dvk

    def madelung(self):
        '''nu_M for exxdiv='ewald', else 0.'''
        if self.exxdiv is None:
            return 0.
        if self.exxdiv != 'ewald':
            raise NotImplementedError(self.exxdiv)
        if getattr(self, '_madelung', None) is None:
            from pyscf.pbc import tools
            self._madelung = tools.pbc.madelung(self.mol, numpy.zeros((1, 3)))
        return self._madelung

    def madelung_energy(self):
        '''-nu_M N / 2 for exxdiv='ewald', else 0 (see the module notes).'''
        return -.5 * self.madelung() * self.mol.nelectron

    def wavefunction_constant(self):
        '''The G = 0 terms as constants of any N-electron wavefunction:
        -nu_M N / 2 and, for lr='gspace' vectors, -c0 N (N - 1) / 2.'''
        e = self.madelung_energy()
        g0 = getattr(self.with_df, 'g0_constant', None)
        if g0 is not None:
            e += g0(self.mol.nelectron)
        return e

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        # vhf from get_veff carries the G = 0 terms explicitly (g0_terms);
        # energy_nuc carries their wavefunction form: take it out
        return super().energy_tot(dm, h1e, vhf) - self.wavefunction_constant()

    def get_ovlp(self, mol=None):
        if mol is None:
            mol = self.mol
        s = numpy.asarray(mol.pbc_intor('int1e_ovlp', hermi=1))
        return spinor_hf.sph2spinor(mol, _sph_blockdiag(s))

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        t = numpy.asarray(mol.pbc_intor('int1e_kin', hermi=1))
        h = spinor_hf.sph2spinor(mol, _sph_blockdiag(t + self.with_df.get_nuc()))
        x2c = getattr(self, 'with_x2c', None)
        if x2c is not None:
            if not isinstance(x2c, OneCenterX2CMPHelper):
                raise TypeError('periodic SCF: with_x2c must be a OneCenterX2CMPHelper')
            h = h + x2c.get_correction()
        return h

    def _attach_x2cmp(self, flavor, with_gaunt, with_breit, with_pcc, with_aoc):
        self.with_x2c = OneCenterX2CMPHelper(self.mol, flavor, with_gaunt, with_breit, with_pcc, with_aoc)
        self._keys = self._keys.union(['with_x2c'])
        return self

    def x2camf(self, with_gaunt=True, with_breit=True, with_pcc=False, with_aoc=False):
        '''X2CAMF: the one-centre approximation of X2CMP (OneCenterX2CMPHelper).'''
        return self._attach_x2cmp('x2camf', with_gaunt, with_breit, with_pcc, with_aoc)

    def x2cmp(self, flavor='x2cmp_1c', with_gaunt=True, with_breit=True, with_pcc=False, with_aoc=False):
        '''X2CMP for a cell: only its one-centre approximation exists, flavor
        'x2cmp_1c' (atomic X2CMP blocks) or 'x2camf'; anything else (the
        molecular-mean-field 'x2cmp', 'sfmp', ...) raises.'''
        return self._attach_x2cmp(flavor, with_gaunt, with_breit, with_pcc, with_aoc)

    def energy_nuc(self):
        return self.with_df.energy_nuc() + self.wavefunction_constant()


def nuclear_zeta(cell):
    '''Gaussian nuclear exponents of the cell's nuclear model (cell.nucmod,
    e.g. 'G' = Visscher-Dyall, the model of the x2camf atomic code), or None
    for point nuclei.  Pass them to RSDF(..., nuc_zeta=nuclear_zeta(cell)).'''
    from pyscf import gto
    z = numpy.array([cell._env[cell._atm[ia, gto.PTR_ZETA]] for ia in range(cell.natm)])
    return None if (z == 0).all() else z


def check_nuclear_model(cell, rsdf, log=None):
    '''RSDF's Gaussian nuclei must be the cell's nuclear model: V_ne, E_nn and
    J come from RSDF, while the one-centre X2C blocks (and h_NR(A) inside them)
    use the cell's nucmod.  A mismatch shows up as a finite-nucleus energy error
    (Kr, point vs Gaussian zeta = 1e12: 6e-6 Eh at the NR level, more with X2C).'''
    import warnings
    zc = nuclear_zeta(cell)
    zr = numpy.broadcast_to(numpy.asarray(rsdf.nuc_zeta, dtype=float), (cell.natm,))
    if zc is None:
        if zr.min() < 1e13:
            warnings.warn('point nuclei in the cell, but RSDF uses Gaussian nuclei with zeta >= %.1e: '
                          'finite-nucleus error ~1e-5 Eh for heavy atoms; use RSDF(..., nuc_zeta=1e14) '
                          'or a Gaussian cell.nucmod with nuc_zeta=nuclear_zeta(cell)' % zr.min())
    elif not numpy.allclose(zr, zc, rtol=1e-10):
        raise ValueError('RSDF nuc_zeta differs from the Gaussian nuclear model of the cell; '
                         'use RSDF(cell, auxbasis, nuc_zeta=spinor_scf.nuclear_zeta(cell))')


def periodic(mf, rsdf):
    '''Turn a molecular socutils spinor SCF object on a cell into the
    Gamma-point periodic one with RSDF Coulomb interactions.'''
    check_nuclear_model(mf.mol, rsdf)
    if rsdf.cderi is None:
        rsdf.build()
    dfmf = _PeriodicSpinorSCF(mf, rsdf, False)
    dfmf.direct_scf = False   # the MO-based K path needs the full (tagged) DM
    return lib.set_class(dfmf, (_PeriodicSpinorSCF, mf.__class__))


def SpinorRSDF(cell, rsdf):
    '''Gamma-point j-adapted spinor HF of a cell (socutils SpinorSCF + RSDF).'''
    return periodic(spinor_hf.SpinorSCF(cell), rsdf)

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
# unchanged, so CASCI/CASSCF on top see the same constant.  exxdiv=None gives
# the bare G = 0-free energy (error ~ 1/L for a molecule in a box).
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
    # Frozen-exchange inner iterations: K is rebuilt every k_rebuild_cycle
    # SCF iterations and kept fixed in between (J is always rebuilt).  The
    # convergence test passes only on an iteration whose K is exact, and the
    # final Fock matrix uses an exact K.  (In an AO basis this is what ACE
    # buys: the ACE operator reproduces the frozen K on the occupied space
    # only, so reusing the full frozen K is at least as accurate.)
    k_rebuild_cycle = 1
    _keys = {'exxdiv', 'k_rebuild_cycle'}

    def _k_reset(self):
        self._k_frozen = None
        self._k_age = 0
        self._k_exact = True
        self._k_force = False
        self._k_post = False
        self._e_exact_last = None
        self.k_builds = 0

    def scf(self, dm0=None, **kwargs):
        self._k_reset()
        return super().scf(dm0, **kwargs)
    kernel = lib.alias(scf, alias_name='kernel')

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if not hasattr(self, '_k_frozen'):
            self._k_reset()
        single = numpy.ndim(dm) == 2
        if (self.k_rebuild_cycle <= 1 or not single or self._k_frozen is None or self._k_force
                or self._k_age + 1 >= self.k_rebuild_cycle):
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            if single:
                self._k_frozen, self._k_age, self._k_exact, self._k_force = vk, 0, True, False
                self.k_builds += 1
        else:
            vj = self.get_jk(mol, dm, hermi=hermi, with_k=False)[0]
            vk = self._k_frozen
            self._k_age += 1
            self._k_exact = False
        return vj - vk

    def check_convergence(self, envs):
        e = envs['e_tot']
        if getattr(self, '_k_post', False):
            # pyscf's extra cycle after convergence (its own, looser test)
            return abs(e - envs['last_hf_e']) < envs['conv_tol'] or envs['norm_gorb'] < envs['conv_tol_grad']
        if not getattr(self, '_k_exact', True):
            return False      # a frozen-K iteration never converges
        # with frozen K in between, consecutive energies differ by the K update:
        # compare with the previous exact-K iteration instead
        e_ref = getattr(self, '_e_exact_last', None)
        e_ref = envs['last_hf_e'] if self.k_rebuild_cycle <= 1 or e_ref is None else e_ref
        self._e_exact_last = e
        conv = abs(e - e_ref) < envs['conv_tol'] and envs['norm_gorb'] < envs['conv_tol_grad']
        if conv:
            self._k_force = True      # the extra cycle uses an exact K
            self._k_post = True
        return conv

    def madelung_energy(self):
        '''-nu_M N / 2 for exxdiv='ewald', else 0 (see the module notes).'''
        if self.exxdiv is None:
            return 0.
        if self.exxdiv != 'ewald':
            raise NotImplementedError(self.exxdiv)
        from pyscf.pbc import tools
        return -.5 * tools.pbc.madelung(self.mol, numpy.zeros((1, 3))) * self.mol.nelectron

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
        return self.with_df.energy_nuc() + self.madelung_energy()


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

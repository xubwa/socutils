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
#     relativity  X2CAMF (mf.x2camf()): range-separated X2C-1e for the cell
#               plus the atomic mean-field SOC/Gaunt/Breit, see
#               RangeSeparatedX2CAMFHelper; or the one-centre approximation of
#               X2CMP, OneCenterX2CMPHelper: mf.x2cmp('x2cmp_1c' or 'x2camf_1c')
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


# one-centre periodic flavors, keyed by the molecular flavor used for the
# isolated atoms
ONE_CENTER_FLAVORS = {'x2camf_1c': 'x2camf',   # one-centre X2C-1e + atomic mean-field SOC
                      'x2cmp_1c': 'x2cmp'}     # atomic X2CMP (atomic Fock decoupled)


class OneCenterX2CMPHelper:
    '''X2CMP in the one-centre approximation for a cell (or any molecule).

    The relativistic correction of the one-electron Hamiltonian is a sum of
    atomic blocks,
        h = h_NR + sum_A [h_X2C(A) - h_NR(A)],
    where h_X2C(A) is the molecular socutils X2C Hamiltonian of the isolated
    atom A in its own basis (flavor 'x2camf_1c': X2C-1e decoupling plus the
    atomic mean-field SOC; 'x2cmp_1c': the atomic 4c Fock matrix decoupled),
    and h_NR(A) = T + V_A of the same atom.  All multi-centre relativistic
    terms are dropped, including the scalar ones: the other nuclei never enter
    the decoupling and no two-centre picture change is made.  This is NOT
    X2CAMF: X2CAMF decouples the whole one-electron Dirac Hamiltonian (all
    nuclei, X2C-1e) and only its two-electron SOC/Gaunt/Breit part is atomic
    (RangeSeparatedX2CAMFHelper is the periodic X2CAMF).  h_NR of the cell (T
    and the periodic V_ne) is supplied by the caller.
    '''
    def __init__(self, mol, flavor='x2camf_1c', with_gaunt=True, with_breit=True,
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


def sr_nuclear_integrals(mol, omega, nuc_zeta=1e14, precision=1e-12):
    '''Gamma-point matrices of the short-range nuclear potential

        V_sr(r) = -sum_{A,T} Z_A erfc(omega |r - R_A - T|) / |r - R_A - T|

    (Gaussian nuclei nuc_zeta, the RSDF model) in the spherical AO basis of
    mol: v = <m|V_sr|n>, pvp = <grad m|V_sr|grad n> and pxp[k] = <grad m x
    V_sr grad n>_k, summed over the AO images T' of n (Gamma point).  V_sr is
    short-ranged, so the lattice sum converges absolutely in real space: no
    G = 0 term, no neutralising background.  For a Mole (no lattice) these are
    the molecular integrals, and omega = 0 gives the full Coulomb potential.'''
    from pyscf import gto
    from socutils.pbc.coulomb_reference import lattice_translations

    def shifted(t):
        # copy of the basis of mol (whatever its _basis, e.g. decontracted) at R + t
        m = gto.Mole()
        m._atm, m._bas, m._env = mol._atm.copy(), mol._bas.copy(), mol._env.copy()
        for ia in range(mol.natm):
            p = m._atm[ia, gto.PTR_COORD]
            m._env[p:p + 3] += t
        m.cart = mol.cart
        m._built = True
        return m
    zeta = numpy.broadcast_to(numpy.asarray(nuc_zeta, dtype=float), (mol.natm,))
    coords = mol.atom_coords()
    lnp = -numpy.log(precision)
    amin = min(mol.bas_exp(i).min() for i in range(mol.nbas))
    r_ao = numpy.sqrt(lnp / amin)                     # AO extent
    r_sr = (numpy.sqrt(lnp) + 1.) / omega if omega else numpy.inf   # erfc(omega r)/r < precision
    if getattr(mol, 'dimension', 0) == 3:
        if not omega:
            raise ValueError('sr_nuclear_integrals: omega = 0 (full Coulomb) only for a molecule')
        a = mol.lattice_vectors()
        span = numpy.linalg.norm(coords[:, None] - coords[None], axis=2).max()
        t_ao = lattice_translations(a, 2 * r_ao + span)
        t_ao = numpy.array([t for t in t_ao
                            if numpy.min(numpy.linalg.norm(coords[:, None] - coords[None] - t, axis=2)) <= 2 * r_ao])
        t_nuc = lattice_translations(a, r_ao + r_sr + span)
    elif getattr(mol, 'dimension', 0) == 0 or not hasattr(mol, 'lattice_vectors'):
        t_ao = t_nuc = numpy.zeros((1, 3))
    else:
        raise NotImplementedError('sr_nuclear_integrals: dimension %d' % mol.dimension)
    # nuclei within reach of the products of cell-0 AOs
    nuc = [(ia, coords[ia] + t) for t in t_nuc for ia in range(mol.natm)
           if numpy.min(numpy.linalg.norm(coords - (coords[ia] + t), axis=1)) <= r_ao + r_sr]
    omega = omega or 0.
    fm = gto.Mole()
    fm.atom = [('X%d' % ia, r) for ia, r in nuc]
    fm.unit = 'Bohr'
    fm.basis = {'X%d' % ia: [[0, [zeta[ia], 1.]]] for ia in range(mol.natm)}
    fm.verbose = 0
    fm.build(False, False)
    ids = numpy.array([ia for ia, _ in nuc])
    q = (numpy.pi / zeta[ids])**1.5 * (2 * zeta[ids] / numpy.pi)**.75   # charge of a libcint s shell
    chg = -mol.atom_charges()[ids] / q
    mol0 = shifted(numpy.zeros(3))
    nao = mol.nao_nr()
    v = numpy.zeros((nao, nao))
    pvp = numpy.zeros((nao, nao))
    pxp = numpy.zeros((3, nao, nao))
    for t in t_ao:
        pm = gto.conc_mol(gto.conc_mol(mol0, shifted(t)), fm)
        pm.omega = -omega
        nb = mol0.nbas
        sl = (0, nb, nb, 2 * nb, 2 * nb, pm.nbas)
        v += numpy.einsum('ijp,p->ij', pm.intor('int3c2e', shls_slice=sl), chg)
        pvp += numpy.einsum('ijp,p->ij', pm.intor('int3c2e_pvp1', shls_slice=sl), chg)
        pxp += numpy.einsum('xijp,p->xij', pm.intor('int3c2e_pvxp1', shls_slice=sl, comp=3), chg)
    return v, pvp, pxp


def _spin_orbital_w(pvp, pxp):
    '''<sigma.p m|V|sigma.p n> in the [alpha | beta] spin-orbital layout of
    spinor_hf.sph2spinor: pVp + i sigma.(p x Vp).'''
    n = pvp.shape[0]
    bx, by, bz = pxp
    w = numpy.zeros((2 * n, 2 * n), dtype=complex)
    w[:n, :n] = pvp + 1j * bz
    w[n:, n:] = pvp - 1j * bz
    w[:n, n:] = 1j * bx + by
    w[n:, :n] = 1j * bx - by
    return w


class RangeSeparatedX2CAMFHelper:
    '''Periodic X2CAMF with a range-separated X2C-1e (rs1e) scalar + 1e SOC part.

        h = h_X2C-1e[V_sr] + (V_ne - V_sr) + sum_A h_AMF(A)

    * h_X2C-1e[V_sr]: the modified Dirac equation of the cell with the short-
      range nuclear potential V_sr = -Z erfc(omega r)/r of every nucleus
      (sr_nuclear_integrals), decoupled at Gamma in the uncontracted basis and
      contracted as in the molecular code (C^T h C).  It carries the scalar
      and one-electron spin-orbit relativity of all nuclei within ~1/omega,
      including the multi-centre terms.
    * V_ne - V_sr: the long-range rest of the periodic (RSDF, G = 0-free)
      nuclear potential, untransformed, like the electron-electron
      interaction.  This is what keeps X2C-1e consistent with the
      untransformed J for a cell: a full X2C-1e decoupling of the Ewald
      potential would picture-change the long-range field of bare nuclei and
      the neutralising background of the nuclei (but not those of the
      electrons), which gives an O(c^-2) volume-dependent error
      -(pi/2c^2) rho per electron (rho = sum Z / Omega) and multipole and
      macroscopic-field terms in low-symmetry sites and slabs.
    * h_AMF(A) = h_X2CAMF(A) - h_X2C-1e(A) of the isolated atom: the atomic
      mean-field two-electron SOC (+ Gaunt, Breit), one-centre, exactly the
      atomic part of the molecular X2CAMF.
    * self_lr (default True): + sum_A d_A, d_A = h_X2C-1e[V_A] - h_X2C-1e[V_sr,A]
      - V_lr,A of the isolated atom (same integrals and nuclear model).  The
      long-range part of an atom's OWN nucleus is smooth at that atom and its
      missing picture change is a Darwin-like term of the Gaussian
      countercharge, ~ (pi/2c^2) Z (omega^2/pi)^1.5 per electron (HI, omega = 1:
      28 mEh in the total energy, ~1 mEh in every orbital energy of I).  It is
      one-centre and transferable, so d_A puts it back exactly: an isolated atom
      is then X2C-1e(+AMF), and only the long-range parts of the OTHER nuclei
      (and the Ewald background) stay untransformed.

    For omega -> 0 (and a molecule) this is the molecular X2CAMF.  omega = 1
    bohr^-1 is the default of the molecular rs1e tests; the Gaussian
    countercharges of neighbouring nuclei must not overlap an atom
    (exp(-omega^2 R_nn^2) << 1), or their sum turns back into a smooth
    background inside the decoupling.
    '''
    def __init__(self, mol, omega=1.0, nuc_zeta=1e14, with_gaunt=True, with_breit=True,
                 with_pcc=False, with_aoc=False, xuncontract=True, precision=1e-12, self_lr=True):
        self.mol = mol
        self.omega = omega
        self.nuc_zeta = nuc_zeta
        self.gaunt = with_gaunt
        self.breit = with_breit
        self.pcc = with_pcc
        self.aoc = with_aoc
        self.xuncontract = xuncontract
        self.precision = precision
        self.self_lr = self_lr
        self.amf = {}         # element -> h_AMF(A) (+ d_A), spinor basis of A
        self._h = None
        self.v_sr = None      # contracted spherical V_sr, for V_ne - V_sr

    def _atom_amf(self, ia):
        from pyscf import gto
        from socutils.somf.x2cmp import SpinorX2CMPHelper
        mol = self.mol
        key = mol.atom_symbol(ia)
        if key not in self.amf:
            atom = gto.Mole()
            atom.atom = [(key, (0., 0., 0.))]
            atom.unit = 'Bohr'
            atom.basis = {key: mol._basis[key]}
            atom.nucmod = {key: mol.nucmod[key]} if isinstance(mol.nucmod, dict) and key in mol.nucmod else {}
            atom.cart = mol.cart
            atom.verbose = 0
            atom.spin = gto.charge(key) % 2
            atom.build(False, False)
            h = {}
            for flavor in ('x2camf', '1e'):
                h[flavor] = SpinorX2CMPHelper(atom, x2cmp=flavor, with_gaunt=self.gaunt, with_breit=self.breit,
                                              with_pcc=self.pcc, with_aoc=self.aoc).get_hcore(atom)
            d = h['x2camf'] - h['1e']
            if self.self_lr:
                zeta = numpy.broadcast_to(numpy.asarray(self.nuc_zeta, dtype=float), (mol.natm,))[ia]
                kw = dict(nuc_zeta=zeta, xuncontract=self.xuncontract, precision=self.precision, self_lr=False)
                h_full, v_full = RangeSeparatedX2CAMFHelper(atom, 0., **kw).get_rs1e()
                h_rs, v_sr = RangeSeparatedX2CAMFHelper(atom, self.omega, **kw).get_rs1e()
                d = d + h_full - h_rs - spinor_hf.sph2spinor(atom, _sph_blockdiag(v_full - v_sr))
            self.amf[key] = d
        return self.amf[key]

    def get_amf(self):
        '''sum_A h_AMF(A) (+ d_A), spinor basis of mol, block diagonal by atom.'''
        mol = self.mol
        n2c = mol.nao_2c()
        d = numpy.zeros((n2c, n2c), dtype=complex)
        for ia, (_, _, p0, p1) in enumerate(mol.aoslice_2c_by_atom()):
            da = self._atom_amf(ia)
            assert da.shape == (p1 - p0, p1 - p0)
            d[p0:p1, p0:p1] = da
        return d

    def get_xmol(self):
        mol = self.mol
        if not self.xuncontract:
            return mol, numpy.eye(mol.nao_nr())
        return mol.decontract_basis(aggregate=True)

    def get_rs1e(self):
        '''h_X2C-1e[V_sr] in the spinor basis of mol and the contracted
        spherical V_sr.'''
        from pyscf.x2c import x2c
        from pyscf.lib import param
        xmol, contr = self.get_xmol()
        if getattr(xmol, 'dimension', 0):
            t = numpy.asarray(xmol.pbc_intor('int1e_kin', hermi=1))
            s = numpy.asarray(xmol.pbc_intor('int1e_ovlp', hermi=1))
        else:
            t = xmol.intor('int1e_kin')
            s = xmol.intor('int1e_ovlp')
        v, pvp, pxp = sr_nuclear_integrals(xmol, self.omega, self.nuc_zeta, self.precision)
        t2, s2, v2 = _sph_blockdiag(t), _sph_blockdiag(s), _sph_blockdiag(v)
        w2 = _spin_orbital_w(pvp, pxp)
        c = param.LIGHT_SPEED
        xm = x2c._x2c1e_xmatrix(t2, v2, w2, s2, c)
        h = x2c._get_hcore_fw(t2, v2, w2, s2, xm, c)
        nx, n = contr.shape
        c2 = numpy.zeros((2 * nx, 2 * n))
        c2[:nx, :n] = c2[nx:, n:] = contr
        h = c2.T @ h @ c2
        return spinor_hf.sph2spinor(self.mol, h), contr.T @ v @ contr

    def get_hcore(self, v_ne):
        '''Full one-electron Hamiltonian (spinor basis) given the periodic
        nuclear attraction v_ne (spherical, contracted basis, e.g. RSDF.get_nuc()).'''
        if self._h is None:
            h_rs, self.v_sr = self.get_rs1e()
            self._h = h_rs + self.get_amf()
        return self._h + spinor_hf.sph2spinor(self.mol, _sph_blockdiag(v_ne - self.v_sr))


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
        x2c = getattr(self, 'with_x2c', None)
        v_ne = self.with_df.get_nuc()
        if isinstance(x2c, RangeSeparatedX2CAMFHelper):
            return x2c.get_hcore(v_ne)
        t = numpy.asarray(mol.pbc_intor('int1e_kin', hermi=1))
        h = spinor_hf.sph2spinor(mol, _sph_blockdiag(t + v_ne))
        if x2c is not None:
            if not isinstance(x2c, OneCenterX2CMPHelper):
                raise TypeError('periodic SCF: with_x2c must be a RangeSeparatedX2CAMFHelper '
                                'or a OneCenterX2CMPHelper')
            h = h + x2c.get_correction()
        return h

    def _attach_x2cmp(self, flavor, with_gaunt, with_breit, with_pcc, with_aoc):
        self.with_x2c = OneCenterX2CMPHelper(self.mol, flavor, with_gaunt, with_breit, with_pcc, with_aoc)
        self._keys = self._keys.union(['with_x2c'])
        return self

    def x2camf(self, with_gaunt=True, with_breit=True, with_pcc=False, with_aoc=False, omega=1.0):
        '''X2CAMF for a cell: range-separated X2C-1e (short-range nuclear
        potential decoupled, omega in bohr^-1) plus the atomic mean-field
        SOC/Gaunt/Breit (RangeSeparatedX2CAMFHelper).'''
        self.with_x2c = RangeSeparatedX2CAMFHelper(self.mol, omega, self.with_df.nuc_zeta, with_gaunt,
                                                   with_breit, with_pcc, with_aoc)
        self._keys = self._keys.union(['with_x2c'])
        return self

    def x2cmp(self, flavor='x2cmp_1c', with_gaunt=True, with_breit=True, with_pcc=False, with_aoc=False):
        '''One-centre approximations for a cell (OneCenterX2CMPHelper): flavor
        'x2cmp_1c' (atomic X2CMP blocks) or 'x2camf_1c' (atomic X2CAMF blocks,
        scalar relativity one-centre as well); anything else (the molecular-
        mean-field 'x2cmp', 'sfmp', ...) raises.  For X2CAMF use mf.x2camf().'''
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

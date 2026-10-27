'''
Compact molecular integration grids for DFT energies, gradients and analytic
Hessians, and the translational sum rule that makes coarse grids usable for
Hessians in PySCF.

Why
---
PySCF's RKS/UKS Hessians (``pyscf.hessian.rks``) treat the grid as fixed in
space (no grid-point / Becke-weight derivatives).  On a finite grid this
breaks translational invariance, and the error lands almost entirely in the
one-centre diagonal blocks H_AA: second derivatives of the tight core basis
functions of atom A integrated on a grid that does not move with A.  With
the default level-2 grid this alone gives frequency errors of ~2-30 cm-1 for
second-row atoms (CH3Cl, H2S) and ~4 cm-1 in the soft modes of (H2O)2; with
an ORCA-sized grid (M3 radial mapping) hundreds of cm-1.  Imposing the
invariance, H_AA = -sum_{B!=A} H_AB (:func:`translational_sum_rule`),
removes this error at no cost and is what allows grids of ~4000-5000
points/atom (comparable to ORCA's DefGrid2) for frequencies.  References:
B. G. Johnson, M. J. Frisch, J. Chem. Phys. 100, 7429 (1994) (quadrature
weight derivatives in DFT second derivatives); H. Shang, arXiv:1801.00166
(the moving-grid effect acts mainly on the diagonal force constants and can
be bypassed through translational symmetry).

Grid ingredients (all from published formulas)
---------------------------------------------
* **Radial grid**: Gauss-Chebyshev (second kind) quadrature with the
  Treutler-Ahlrichs M3 (or M4) mapping,

      M3:  r = xi / ln2 * ln(2 / (1 - x))
      M4:  r = xi / ln2 * (1 + x)^alpha * ln(2 / (1 - x)),  alpha = 0.6

  O. Treutler and R. Ahlrichs, J. Chem. Phys. 102, 346 (1995),
  DOI:10.1063/1.469408, with their element parameters ``xi`` (Table I, as
  tabulated in ``pyscf.dft.radi``).  M3 is used for H-Ne: it keeps more
  shells in the bonding region than M4 / Mura-Knowles and gives the most
  accurate (sum-rule) Hessians per point.  For Z > 10 the M3 core is too
  coarse for the energy, so the Mura-Knowles log3 grid (M. E. Mura,
  P. J. Knowles, J. Chem. Phys. 104, 9848 (1996)) is used there.

* **Number of radial shells** from an accuracy parameter ``intacc``
  (M. Krack, A. M. Koester, J. Chem. Phys. 108, 3226 (1998)):

      n_rad = round(15 * intacc - 40 + b * ROW)

  ROW is the period; ``b`` = 5 (the original value) for H-Ne and
  ``radial_b_heavy`` = 7.5 for Z > 10 (fitted here to H2S / CH3Cl energies).
  With intacc = 4.388: H 31, C/O 36, S/Cl 48 shells.

* **Angular pruning**: five radial regions with different Lebedev grids
  (SG-1 / NWChem style: P. M. W. Gill, B. G. Johnson, J. A. Pople,
  Chem. Phys. Lett. 209, 506 (1993)).  ``angular_grid = 1..7`` selects one
  of the five-region tables of the ORCA 6.0 manual (Table 7.1, e.g. 4 =
  26/110/194/302/194); the region boundaries (``alphas``) are 1.15 x the
  NWChem multiples of the Bragg radius.  ``h_reduced`` uses the next smaller table
  for H and He.

* **Partition**: Becke cells with the Treutler size adjustment (PySCF's
  default).  Stratmann-Scuseria-Frisch (``gen_grid.stratmann``) and LKO
  (``gen_grid.becke_lko``) are available but were measured to be no better
  for Hessians (LKO and Becke without size adjustment converge to a
  frequency limit ~0.5 cm-1 off for O-H stretches on very fine grids).

* **Weight cutoff**: points with |w| < ``weight_cutoff`` are dropped.

Defaults (``Grids(mol)``, also :func:`hessian_grids`): intacc 4.388,
angular_grid 4, h_reduced.  See dft/test/test_grids.py.  Nothing was taken
from closed-source code.
'''

import numpy
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.dft import gen_grid, radi
from pyscf.dft.LebedevGrid import MakeAngularGrid, LEBEDEV_NGRID

# Five-region Lebedev tables (number of points), regions ordered from the
# nucleus outward.  ORCA 6.0 manual, Table 7.1 ("AngularGrid" 1-7).
ANGULAR_TABLES = {
    1: (14, 26, 50, 50, 26),
    2: (14, 26, 50, 110, 50),
    3: (26, 50, 110, 194, 110),
    4: (26, 110, 194, 302, 194),
    5: (26, 194, 302, 434, 302),
    6: (50, 302, 434, 590, 434),
    7: (110, 434, 590, 770, 590),
}

# Region boundaries in units of the Bragg-Slater radius (rows: H-He, Li-Ne,
# Na and beyond).  NWChem / SG-1 values, as in pyscf.dft.gen_grid.nwchem_prune.
NWCHEM_ALPHAS = numpy.array((
    (0.25, 0.5, 1.0, 4.5),
    (0.1667, 0.5, 0.9, 3.5),
    (0.1, 0.4, 0.8, 2.5)))


def _period(chg):
    return int((chg > numpy.array((2, 10, 18, 36, 54, 86, 118))).sum()) + 1


def _xi(chg):
    if chg < len(radi._treutler_ahlrichs_xi):
        return radi._treutler_ahlrichs_xi[chg]
    return 1.5


def treutler_m3(n, chg, *args, **kwargs):
    '''Treutler-Ahlrichs M3 mapping on Gauss-Chebyshev (2nd kind) nodes.
    JCP 102, 346 (1995); DOI:10.1063/1.469408, eq. (18).'''
    xi = kwargs.get('xi', None)
    if xi is None:
        xi = _xi(chg)
    step = numpy.pi / (n + 1)
    t = numpy.arange(1, n + 1) * step
    x = numpy.cos(t)
    c = xi / numpy.log(2)
    r = c * numpy.log(2 / (1 - x))
    # dx = sin^2(t) * step / sqrt(1-x^2) = sin(t) * step
    dr = step * numpy.sin(t) * c / (1 - x)
    return r[::-1].copy(), dr[::-1].copy()


def treutler_m4(n, chg, *args, **kwargs):
    '''Treutler-Ahlrichs M4 mapping (alpha=0.6) on Gauss-Chebyshev (2nd kind)
    nodes.  JCP 102, 346 (1995); DOI:10.1063/1.469408, eq. (19).'''
    xi = kwargs.get('xi', None)
    if xi is None:
        xi = _xi(chg)
    alpha = kwargs.get('alpha', 0.6)
    step = numpy.pi / (n + 1)
    t = numpy.arange(1, n + 1) * step
    x = numpy.cos(t)
    c = xi / numpy.log(2)
    lg = numpy.log(2 / (1 - x))
    r = c * (1 + x)**alpha * lg
    dr = step * numpy.sin(t) * c * (1 + x)**alpha * (alpha / (1 + x) * lg + 1 / (1 - x))
    return r[::-1].copy(), dr[::-1].copy()


def krack_koster_nrad(chg, intacc, b=5.):
    '''Number of radial shells, n = 15*intacc - 40 + b*ROW
    (Krack, Koester, JCP 108, 3226 (1998))'''
    return int(round(15 * intacc - 40 + b * _period(chg)))


def region_prune(chg, rads, angular_grid, alphas=NWCHEM_ALPHAS, radii=radi.BRAGG_RADII):
    '''Five-region angular pruning.  ``angular_grid`` is a key of
    ANGULAR_TABLES (or a 5-tuple of Lebedev sizes).'''
    if isinstance(angular_grid, (int, numpy.integer)):
        table = numpy.asarray(ANGULAR_TABLES[int(angular_grid)])
    else:
        table = numpy.asarray(angular_grid)
    alphas = numpy.asarray(alphas)
    if alphas.ndim == 2:
        row = 0 if chg <= 2 else (1 if chg <= 10 else 2)
        alphas = alphas[row]
    r_atom = radii[chg] + 1e-200
    place = ((rads / r_atom).reshape(-1, 1) > alphas).sum(axis=1)
    return table[place]


class Grids(gen_grid.Grids):
    '''Compact DFT grid (see module docstring).

    Attributes (beyond pyscf's Grids):
        intacc : float
            Accuracy parameter of the radial grid, n_rad = 15*intacc-40+b*ROW.
        radial_b : float
            b in the radial-shell formula (H-Ne).
        angular_grid : int (1-7)
            Five-region Lebedev table (ANGULAR_TABLES).
        h_reduced : bool
            Use angular_grid-1 for H and He.
        alphas : array (3,4) or (4,)
            Region boundaries in units of the Bragg radius; default 1.15 x
            the NWChem values (2.5% fewer points than the NWChem boundaries
            at equal measured Hessian / energy accuracy).
        weight_cutoff : float
            Drop points with |w| below this value after partitioning.
        element_radi : dict {symbol: radial function}
            Element-specific radial grid (default ``radi_method``).
        xi : dict {symbol: xi}
            Overrides the Treutler-Ahlrichs scaling parameters of the M3/M4
            mappings; ``xi_scale`` multiplies all of them.
        radi_method : radial grid for H-Ne, M3 (``treutler_m3``) by default.
        heavy_radi_method : radial grid for Z > 10, ``radi.mura_knowles`` by
            default (None: use ``radi_method`` for all elements).
        radial_b_heavy : b of the radial-shell formula for Z > 10.
        alpha : exponent of the M4 mapping (``treutler_m4``).
        becke_scheme, radii_adjust : PySCF's Becke partition with the
            Treutler size adjustment by default.

    ``atom_grid`` still overrides the radial / angular sizes element by
    element (then the five-region pruning is applied with the given maximum
    angular size).
    '''
    intacc = 4.388
    radial_b = 5.
    angular_grid = 4
    h_reduced = True
    alphas = NWCHEM_ALPHAS * 1.15
    weight_cutoff = 1e-14
    xi = {}
    xi_scale = 1.
    element_radi = {}
    alpha = 0.6
    radial_b_heavy = 7.5
    radi_method = staticmethod(treutler_m3)
    heavy_radi_method = staticmethod(radi.mura_knowles)
    becke_scheme = staticmethod(gen_grid.original_becke)
    radii_adjust = staticmethod(radi.treutler_atomic_radii_adjust)

    _keys = {'intacc', 'radial_b', 'angular_grid', 'h_reduced', 'alphas', 'weight_cutoff', 'xi', 'xi_scale',
             'element_radi', 'alpha', 'radial_b_heavy', 'heavy_radi_method'}

    def __setattr__(self, key, val):
        if key in ('intacc', 'radial_b', 'angular_grid', 'h_reduced', 'alphas', 'weight_cutoff',
                   'xi', 'xi_scale', 'element_radi', 'alpha', 'radial_b_heavy', 'heavy_radi_method'):
            self.reset()
        super().__setattr__(key, val)

    def dump_flags(self, verbose=None):
        gen_grid.Grids.dump_flags(self, verbose)
        logger.info(self, 'intacc %s  radial_b %s  angular_grid %s  h_reduced %s  weight_cutoff %g',
                    self.intacc, self.radial_b, self.angular_grid, self.h_reduced, self.weight_cutoff)
        return self

    def _atom_setting(self, chg, symb):
        ag = self.angular_grid
        if self.h_reduced and chg <= 2 and isinstance(ag, (int, numpy.integer)):
            ag = max(1, ag - 1)
        atom_grid = self.atom_grid
        if isinstance(atom_grid, (list, tuple)):
            atom_grid = {symb: atom_grid}
        conf = atom_grid.get(symb, atom_grid.get('default', None)) if atom_grid else None
        if conf is not None:
            n_rad, n_ang = conf
            if n_ang not in LEBEDEV_NGRID:
                raise ValueError('Unsupported angular grid %d' % n_ang)
            table = numpy.asarray(ANGULAR_TABLES[ag] if isinstance(ag, (int, numpy.integer)) else ag)
            # scale the five-region table to the requested maximum
            table = numpy.array([min(n, n_ang) for n in table])
            table[table.argmax()] = n_ang
            return n_rad, table
        b = self.radial_b if chg <= 10 else self.radial_b_heavy
        return krack_koster_nrad(chg, self.intacc, b), ag

    def gen_atomic_grids(self, mol, atom_grid=None, radi_method=None, level=None, prune=None, **kwargs):
        if radi_method is None:
            radi_method = self.radi_method
        default_radi = radi_method
        atom_grids_tab = {}
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            if symb in atom_grids_tab:
                continue
            chg = gto.charge(symb)
            n_rad, ag = self._atom_setting(chg, symb)
            if symb in self.element_radi:
                radi_method = self.element_radi[symb]
            elif chg > 10 and self.heavy_radi_method is not None:
                radi_method = self.heavy_radi_method
            else:
                radi_method = default_radi
            kw = dict(kwargs)
            if radi_method in (treutler_m3, treutler_m4):
                kw['xi'] = self.xi.get(symb, _xi(chg)) * self.xi_scale
            if radi_method is treutler_m4:
                kw['alpha'] = self.alpha
            rad, dr = radi_method(n_rad, chg, ia, **kw)
            rad_weight = 4 * numpy.pi * rad**2 * dr
            if self.prune is None:
                n_ang = max(ANGULAR_TABLES[ag]) if isinstance(ag, (int, numpy.integer)) else max(ag)
                angs = numpy.full(n_rad, n_ang)
            else:
                angs = region_prune(chg, rad, ag, self.alphas)
            logger.debug(mol, 'atom %s rad-grids = %d, ang-grids = %s', symb, n_rad, angs)
            coords = []
            vol = []
            for n in sorted(set(angs)):
                grid = MakeAngularGrid(n)
                idx = numpy.where(angs == n)[0]
                coords.append(numpy.einsum('i,jk->jik', rad[idx], grid[:, :3]).reshape(-1, 3))
                vol.append(numpy.einsum('i,j->ji', rad_weight[idx], grid[:, 3]).ravel())
            atom_grids_tab[symb] = (numpy.vstack(coords), numpy.hstack(vol))
        return atom_grids_tab

    def build(self, mol=None, with_non0tab=False, sort_grids=True, **kwargs):
        if mol is None:
            mol = self.mol
        gen_grid.Grids.build(self, mol, with_non0tab=False, sort_grids=sort_grids, **kwargs)
        # drop the padding and the points of negligible weight, then re-pad
        idx = numpy.abs(self.weights) > self.weight_cutoff
        idx &= self.atm_idx >= 0
        self._select_grids(idx)
        self._add_padding()
        if with_non0tab:
            self.non0tab = self.make_mask(mol, self.coords)
            self.screen_index = self.non0tab
        else:
            self.screen_index = self.non0tab = None
        logger.info(self, 'tot grids after weight cutoff = %d', len(self.weights))
        return self


def hessian_grids(mol, **kwargs):
    '''The grid recommended for analytic Hessians (see the module docstring
    and dft/test/test_grids.py).  Keyword arguments override attributes.'''
    g = Grids(mol)
    for k, v in kwargs.items():
        setattr(g, k, v)
    return g


def translational_sum_rule(hess):
    '''Impose translational invariance on a nuclear Hessian (natm,natm,3,3):
    H_AA = -sum_{B!=A} H_AB, symmetrised.

    A molecular grid made of atom-centred grids moves with the nuclei, so the
    exact derivative of the quadrature energy (including the grid-point and
    Becke-weight derivatives) is translationally invariant.  PySCF's RKS/UKS
    Hessians omit those "grid response" terms; on a coarse grid the omission
    shows up almost entirely in the one-centre diagonal blocks H_AA (second
    derivatives of the tight core basis functions on a fixed grid), which can
    be replaced through the invariance condition.  See H. Shang,
    arXiv:1801.00166 ("The moving-grid effect in density functional
    calculations of harmonic vibration frequencies using numeric atom-centered
    grids"), and B. G. Johnson, M. J. Frisch, J. Chem. Phys. 100, 7429 (1994)
    for the weight-derivative terms.  In the converged-grid limit the
    correction vanishes.
    '''
    h = numpy.array(hess, copy=True)
    natm = h.shape[0]
    for a in range(natm):
        h[a, a] = 0
        haa = -h[a].sum(axis=0)
        h[a, a] = (haa + haa.T) * .5
    return h


def hessian(mf, sum_rule=True, cpscf_grids=None, hess_grids=None):
    '''Analytic RKS/UKS Hessian of ``mf`` with the grid options studied here.

    Args:
        sum_rule : apply :func:`translational_sum_rule` (recommended whenever
            the SCF grid is coarser than PySCF level ~4; it costs nothing).
        cpscf_grids : a (coarser) Grids object used only for the XC response
            kernel inside the CP-SCF equations; the SCF grid is used for the
            explicit second-derivative terms and the right-hand sides.
        hess_grids : Grids for the explicit second-derivative terms
            (``partial_hess_elec`` / ``make_h1``); default: the SCF grid.

    Returns:
        Hessian array (natm, natm, 3, 3).
    '''
    hobj = mf.Hessian()
    if hess_grids is not None:
        hobj.grids = hess_grids
    if cpscf_grids is not None:
        if hobj.grids is None:
            hobj.grids = mf.grids
        if cpscf_grids.coords is None:
            cpscf_grids.build(with_non0tab=True)
        solve_mo1 = hobj.solve_mo1
        grids0 = mf.grids

        def solve_mo1_cpscf_grid(*args, **kwargs):
            mf.grids = cpscf_grids
            try:
                return solve_mo1(*args, **kwargs)
            finally:
                mf.grids = grids0
        hobj.solve_mo1 = solve_mo1_cpscf_grid
    h = hobj.kernel()
    if sum_rule:
        h = translational_sum_rule(h)
    return h

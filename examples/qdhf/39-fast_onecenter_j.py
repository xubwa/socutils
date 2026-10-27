#!/usr/bin/env python

'''
Fast, numerically exact Coulomb matrix of superposed spherical atomic densities
(the J part of the one-center model potential of 38-onecenter_model_potential.py),
and its cost against the exact molecular 4c J build x2cmp does and against the
X2C decoupling itself.

Two variants, both reported:
  'erf' : J_K(r) = N_K erf(sqrt(z0) r)/r + V_sr,K(r), N_K = Z_K, V_sr on a
          (150, 590) grid;
  'fit' : J_K(r) = sum_k q_k erf(sqrt(z_k) r)/r + V_res,K(r), the q_k a
          charge-conserving Coulomb fit of rho_K (even-tempered s Gaussians,
          ratio 2.5), the small residual V_res = V_H - V_fit on a coarse
          grid: (75, 194) up to Ar, (150, 302) for heavier elements.  The analytic part carries ~99.99 % of V_H, so the
          Becke-partitioning error of the grid shrinks accordingly.
Grids: pruned, Stratmann partitioning (Becke partitioning is ~100x worse for
these atom-centred integrands in polyatomics).

Steps:

 1. radial (once per element): rho_K(r) from the atomic spinor density blocks
    with real scalar AOs (rho_L = phi^T P phi, rho_S = sum_jk d_j phi^T Q^jk
    d_k phi), 6-point angular average (the atomic density is spherical),
    Hartree potential by cumulative Simpson on a log grid;
 2. analytic part as the attraction to Gaussian nuclei with the sign flipped,
    'erf': one libcint call each for LL and SS (all atoms, N_K = Z_K; 4c and
    2c share LL); 'fit': one call per element and fit exponent, all atoms of
    the element together, charges q_k applied afterwards (4c and 2c share the
    LL integrals);
 3. short range: sum_K V_sr,K on one pruned molecular Becke grid, non0tab
    screening, 6 real dgemms per block (LL for the 4c and 2c densities;
    SS = sum_ij <d_i phi|V|d_j phi> s_i s_j: W0 and the three antisymmetric
    parts, using A_ji = A_ij^T); spinor transformation once at the end.

Printed per molecule: timings of each step, of one nonrelativistic direct JK
build in the same (uncontracted, scalar) basis with the direct-SCF screening
object prebuilt (what an NR SCF iteration costs; the yardstick), of exact J (pyscf DHF get_jk with
the atomic densities, J only, + the 2c J), of the X2C decoupling, the error of
the fast J against exact J (max matrix element), and dE: the SCF energy with
h = to_2c(X(h4 + J4), h4 + J4) - J2 using the fast J minus the same with exact J.

Usage:  python 39-fast_onecenter_j.py [small|large]
  large: chloro-/bromobenzene, 1,4-dibromobenzene, 1-bromonaphthalene; no exact
  4c J and no SCF: accuracy from the 2c J only, whose exact value is one scalar
  J build, J2 = sph2spinor(J_scalar[P]), P = sum_x C_x^* D2 C_x^T
'''

import os
import sys
import time
import contextlib
import numpy
import scipy.linalg
import scipy.special
import scipy.interpolate
from scipy.integrate import cumulative_simpson
from pyscf import gto, scf, lib
lib.einsum = lambda s, *o, **k: numpy.einsum(s, *o, optimize=True)
numpy.linalg.linalg = numpy.linalg
import socutils
from pyscf.lib import param
from pyscf.dft import numint, gen_grid
from pyscf.dft.LebedevGrid import MakeAngularGrid
from pyscf.x2c.x2c import _decontract_spinor
from socutils.scf import spinor_hf
from socutils.somf import x2cmp as X
from x2camf.x2camf import construct_molecular_matrix

os.environ['X2CAMF_TIMING'] = '0'
_PAULI = numpy.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


def quiet(f):
    with open(os.devnull, 'w') as fn, contextlib.redirect_stdout(fn):
        return f()


def atom_mol(xmol, ia):
    symb = xmol.atom_symbol(ia)
    return gto.Mole(atom=[(symb, (0., 0., 0.))], basis={symb: xmol._basis[symb]}, unit='bohr',
                    verbose=0, spin=xmol.atom_charge(ia) % 2).build()


def radial_density(amol, dl, ds=None, nrad=6000, rmin=1e-7, rmax=2e3, nang=6):
    '''4 pi r^2 rho(r) of a spherical atomic density from its spinor blocks
    (dl: LL; ds: SS in the pyscf DHF convention sigma.p chi / 2c, or None)'''
    r = numpy.exp(numpy.linspace(numpy.log(rmin), numpy.log(rmax), nrad))
    ang = MakeAngularGrid(nang)
    cs = amol.sph2spinor_coeff()
    p = sum((cs[x].conj() @ dl @ cs[x].T) for x in (0, 1)).real
    if ds is not None:
        c = param.LIGHT_SPEED
        pyy = [[cs[y].conj() @ ds @ cs[yp].T for yp in (0, 1)] for y in (0, 1)]
        q = numpy.empty((3, 3) + p.shape)
        for j in range(3):
            for k in range(3):
                sjsk = _PAULI[j] @ _PAULI[k]
                q[j, k] = sum(sjsk[y, yp] * pyy[y][yp] for y in (0, 1) for yp in (0, 1)).real * (.25 / c**2)
    rho = numpy.zeros(nrad)
    for xyz, w in zip(ang[:, :3], ang[:, 3]):
        ao = amol.eval_gto('GTOval_sph_deriv1' if ds is not None else 'GTOval_sph', r[:, None] * xyz[None, :])
        if ds is None:
            rho += w * numpy.einsum('gm,gm->g', ao @ p, ao)
        else:
            rho += w * numpy.einsum('gm,gm->g', ao[0] @ p, ao[0])
            for j in range(3):
                for k in range(3):
                    rho += w * numpy.einsum('gm,gm->g', ao[j + 1] @ q[j, k], ao[k + 1])
    return r, 4 * numpy.pi * rho * r**2


def hartree(r, rad):
    '''V_H(r) of the spherical density rad = 4 pi r^2 rho (cumulative Simpson in ln r) and N'''
    lnr = numpy.log(r)
    c1 = cumulative_simpson(rad * r, x=lnr, initial=0.)
    c2 = cumulative_simpson(rad, x=lnr, initial=0.)
    return c1 / r + (c2[-1] - c2), c1[-1]


def coulomb_fit(r, rad, zetas):
    '''charges q_k of normalized s Gaussians (exponents zetas) fitting rad in the
    Coulomb metric, sum q_k = N'''
    lnr = numpy.log(r)
    integ = lambda f: numpy.trapezoid(f * r, lnr)
    a = numpy.asarray(zetas)
    m = 2 / numpy.sqrt(numpy.pi) * numpy.sqrt(numpy.outer(a, a) / numpy.add.outer(a, a))
    b = numpy.array([integ(rad * scipy.special.erf(numpy.sqrt(x) * r) / r) for x in a])
    nk = len(a)
    kkt = numpy.zeros((nk + 1, nk + 1))
    kkt[:nk, :nk] = m
    kkt[:nk, nk] = kkt[nk, :nk] = 1
    return numpy.linalg.lstsq(kkt, numpy.append(b, integ(rad)), rcond=1e-14)[0][:nk]


def element_potentials(xmol, d4, d2, zeta0=1.0, fit_ratio=None):
    '''per element: (spline of the grid part for the 4c density, for the 2c
    density, r_max, analytic part).  Analytic part: fit_ratio None -> N erf(z0)/r
    (returned as zetas [z0], q4 = q2 = [N]); else the Coulomb-fit charges.'''
    n2c = xmol.nao_2c()
    sl = xmol.aoslice_2c_by_atom()
    out = {}
    for ia in range(xmol.natm):
        symb = xmol.atom_symbol(ia)
        if symb in out:
            continue
        i2 = numpy.arange(*sl[ia, 2:4])
        i4 = numpy.concatenate([i2, i2 + n2c])
        nk = len(i2)
        b4 = d4[numpy.ix_(i4, i4)]
        am = atom_mol(xmol, ia)
        r, rad4 = radial_density(am, b4[:nk, :nk], b4[nk:, nk:])
        _, rad2 = radial_density(am, d2[numpy.ix_(i2, i2)])
        if fit_ratio is None:
            z = numpy.array([zeta0])
            qs = [numpy.array([hartree(r, rad)[1]]) for rad in (rad4, rad2)]
        else:
            ex = numpy.concatenate([am.bas_exp(i) for i in range(am.nbas)])
            lo, hi = 2 * ex.min(), 8 * ex.max()
            z = lo * fit_ratio**numpy.arange(int(numpy.ceil(numpy.log(hi / lo) / numpy.log(fit_ratio))) + 1)
            qs = [coulomb_fit(r, rad, z) for rad in (rad4, rad2)]
        spl = []
        for rad, q in zip((rad4, rad2), qs):
            v = hartree(r, rad)[0]
            v -= sum(qk * scipy.special.erf(numpy.sqrt(zk) * r) / r for qk, zk in zip(q, z))
            spl.append(scipy.interpolate.CubicSpline(numpy.log(r), v))
        out[symb] = (spl[0], spl[1], r[-1], (z, qs[0], qs[1]))
    return out


def j_analytic(xmol, pots):
    '''sum_K sum_k q_k erf(sqrt(z_k) r_K)/r_K as LL (4c and 2c charges) and SS
    (sigma.p V sigma.p / 4c^2): attraction to Gaussian nuclei, sign flipped; one
    libcint call per element and exponent, covering all atoms of the element'''
    from pyscf.gto.mole import CHARGE_OF
    c = param.LIGHT_SPEED
    n2c = xmol.nao_2c()
    ll4 = numpy.zeros((n2c, n2c), complex)
    ll2 = numpy.zeros_like(ll4)
    ss4 = numpy.zeros_like(ll4)
    symbs = [xmol.atom_symbol(ia) for ia in range(xmol.natm)]
    for symb, (_, _, _, (z, q4, q2)) in pots.items():
        m = xmol.copy()
        zz = 0
        for ia in range(m.natm):
            if symbs[ia] != symb:
                m._atm[ia, CHARGE_OF] = 0
            else:
                zz = m.atom_charge(ia)
        for zk, a4, a2 in zip(z, q4, q2):
            for ia in range(m.natm):
                m.set_nuc_mod(ia, zk)
            v = -m.intor('int1e_nuc_spinor') / zz
            ll4 += a4 * v
            ll2 += a2 * v
            ss4 -= a4 / zz * m.intor('int1e_spnucsp_spinor') * (.25 / c**2)
    return ll4, ll2, ss4


def j_short_range(xmol, pots, grid):
    g = gen_grid.Grids(xmol)
    g.atom_grid = grid
    g.becke_scheme = gen_grid.stratmann
    g.build(with_non0tab=True)
    ni = numint.NumInt()
    nao = xmol.nao
    ao_loc = xmol.ao_loc_nr()
    shls = (0, xmol.nbas)
    acc = numpy.zeros((6, nao, nao))      # LL4, LL2, W0, A_xy, A_yz, A_zx
    atoms = [(xmol.atom_coord(ia),) + pots[xmol.atom_symbol(ia)][:3] for ia in range(xmol.natm)]
    for ao, mask, weight, coords in ni.block_loop(xmol, g, nao, deriv=1):
        v4 = numpy.zeros(len(weight))
        v2 = numpy.zeros(len(weight))
        for rk, s4, s2, rmax in atoms:
            d = numpy.linalg.norm(coords - rk, axis=1)
            ins = d < rmax
            x = numpy.log(numpy.maximum(d[ins], 1e-7))
            v4[ins] += s4(x)
            v2[ins] += s2(x)
        dot = lambda a, b: numint._dot_ao_ao(xmol, a, b, mask, shls, ao_loc)
        w4 = (weight * v4)[:, None]
        acc[0] += dot(ao[0], ao[0] * w4)
        acc[1] += dot(ao[0], ao[0] * (weight * v2)[:, None])
        wd = [ao[i] * w4 for i in (1, 2, 3)]
        acc[2] += dot(ao[1], wd[0]) + dot(ao[2], wd[1]) + dot(ao[3], wd[2])
        acc[3] += dot(ao[1], wd[1])
        acc[4] += dot(ao[2], wd[2])
        acc[5] += dot(ao[3], wd[0])
    ll4, ll2, w0, axy, ayz, azx = acc
    wx, wy, wz = ayz - ayz.T, azx - azx.T, axy - axy.T
    ca, cb = xmol.sph2spinor_coeff()
    sc = lambda m: ca.conj().T @ m @ ca + cb.conj().T @ m @ cb
    ss = (ca.conj().T @ (w0 + 1j * wz) @ ca + ca.conj().T @ (1j * wx + wy) @ cb
          + cb.conj().T @ (1j * wx - wy) @ ca + cb.conj().T @ (w0 - 1j * wz) @ cb)
    c = param.LIGHT_SPEED
    return sc(ll4), sc(ll2), ss * (.25 / c**2), len(g.weights)


def j_fast(xmol, d4, d2, method='fit'):
    '''method 'erf' (N erf/r analytic, (150, 590) grid) or 'fit' (Coulomb fit
    analytic, ratio 2.5; residual on a (75, 194) grid).  Returns J4, J2, timings
    (radial, analytic, grid), number of grid points.'''
    if method == 'erf':
        fit_ratio, grid = None, (150, 590)
    else:
        # residual grid per element: (75, 194) up to Ar, (150, 302) beyond (Br needs
        # 150 radial points for the SS block: dE 2.5e-8 at 75, 9e-11 at 150)
        fit_ratio = 2.5
        grid = {xmol.atom_symbol(ia): (75, 194) if xmol.atom_charge(ia) <= 18 else (150, 302)
                for ia in range(xmol.natm)}
    t = [time.perf_counter()]
    pots = element_potentials(xmol, d4, d2, 1.0, fit_ratio)
    t.append(time.perf_counter())
    a_ll4, a_ll2, a_ss = j_analytic(xmol, pots)
    t.append(time.perf_counter())
    ll4, ll2, ss, ng = j_short_range(xmol, pots, grid)
    t.append(time.perf_counter())
    j4 = scipy.linalg.block_diag(a_ll4 + ll4, a_ss + ss)
    return j4, a_ll2 + ll2, numpy.diff(t), ng


systems = [
    ('HCl', 'H 0 0 0; Cl 0 0 1.275', 'unc-cc-pvdz'),
    ('HBr', 'H 0 0 0; Br 0 0 1.414', 'unc-cc-pvdz-dk'),
    ('CH3Cl', 'C 0 0 0; Cl 0 0 1.785; H 1.027 0 -0.36; H -0.513 0.889 -0.36; H -0.513 -0.889 -0.36', 'unc-cc-pvdz'),
    ('CH2Cl2', 'C 0 0 0.77; Cl 1.476 0 -0.2; Cl -1.476 0 -0.2; H 0 0.895 1.39; H 0 -0.895 1.39', 'unc-cc-pvdz'),
    ('CHCl3', 'C 0 0 0.45; H 0 0 1.52; Cl 0 1.68 -0.12; Cl 1.455 -0.84 -0.12; Cl -1.455 -0.84 -0.12', 'unc-cc-pvdz'),
    ('CH2Br2', 'C 0 0 0.94; Br 1.59 0 -0.18; Br -1.59 0 -0.18; H 0 0.9 1.55; H 0 -0.9 1.55', 'unc-cc-pvdz-dk'),
]


def aromatic(ring_x, subst):
    '''benzene (ring_x = 1) or naphthalene (ring_x = 2) in the xy plane; the
    outer C-H positions listed in subst are replaced: {index: (symbol, bond / A)}'''
    cc, ch = 1.39, 1.08
    h = cc * numpy.sqrt(3) / 2
    if ring_x == 1:
        ang = numpy.arange(6) * numpy.pi / 3
        cs = numpy.c_[cc * numpy.cos(ang), cc * numpy.sin(ang), numpy.zeros(6)]
        outer = list(range(6))
        centre = lambda x: numpy.zeros(3)
    else:
        cs = numpy.array([[0, cc / 2, 0], [0, -cc / 2, 0], [h, cc, 0], [2 * h, cc / 2, 0], [2 * h, -cc / 2, 0],
                          [h, -cc, 0], [-h, -cc, 0], [-2 * h, -cc / 2, 0], [-2 * h, cc / 2, 0], [-h, cc, 0]])
        outer = [2, 3, 4, 5, 6, 7, 8, 9]
        centre = lambda x: numpy.array([numpy.sign(x[0]) * h, 0., 0.])
    atoms = ['C %.4f %.4f %.4f' % tuple(x) for x in cs]
    for n, i in enumerate(outer):
        d = cs[i] - centre(cs[i])
        d /= numpy.linalg.norm(d)
        symb, r = subst.get(n, ('H', ch))
        atoms.append('%s %.4f %.4f %.4f' % ((symb,) + tuple(cs[i] + r * d)))
    return '; '.join(atoms)

if len(sys.argv) > 1 and sys.argv[1] == 'large':
    systems = [
        ('PhCl', aromatic(1, {0: ('Cl', 1.74)}), 'unc-cc-pvdz'),
        ('PhBr', aromatic(1, {0: ('Br', 1.90)}), 'unc-cc-pvdz-dk'),
        ('C6H4Br2', aromatic(1, {0: ('Br', 1.90), 3: ('Br', 1.90)}), 'unc-cc-pvdz-dk'),
        ('C10H7Br', aromatic(2, {0: ('Br', 1.90)}), 'unc-cc-pvdz-dk'),
    ]

large = len(sys.argv) > 1 and sys.argv[1] == 'large'
print('# fast one-center J; times in s; NR J(K): one scalar direct J (JK) build, same uncontracted basis')
if large:
    print('# exact 2c J from one scalar J build; dJ2 = J2(fast) - J2(exact)')
    print('%-7s %5s %4s | %6s %6s %6s | %-4s %5s %6s %6s %6s %6s %7s | %9s %9s' % (
        'mol', 'n2c', 'natm', 'NR J', 'NR JK', 'X2C', 'J', 'ngrid', 'radial', 'analyt', 'grid', 'total',
        '/NR JK', 'max dJ2', 'TrD2 dJ2'))
else:
    print('# dE = E(fast J) - E(exact J), Eh')
    print('%-7s %5s | %6s %8s %6s | %-4s %5s %6s %6s %6s %6s %6s | %9s %9s %9s %9s' % (
        'mol', 'n2c', 'NR JK', 'exact J', 'X2C', 'J', 'ngrid', 'radial', 'analyt', 'grid', 'total', '/NR JK',
        'max dJ LL', 'max dJ SS', 'max dh', 'dE'))
for name, geom, base in systems:
    mol = gto.M(atom=geom, basis=base[4:], verbose=0)
    mol = gto.M(atom=geom, basis={e: gto.mole.uncontracted_basis(gto.basis.load(base[4:], e))
                                  for e in set(a[0] for a in mol._atom)}, verbose=0)
    mf = spinor_hf.SCF(mol)
    mf.x2cmp(flavor='x2cmp', with_gaunt=False, with_breit=False, with_pcc=False)
    xo = mf.with_x2c
    if large:
        # only the atomic densities are needed; skip the molecular 4c work of x2cmp
        from socutils.somf.x2cmp import x2camf as _x2camf_mod
        ints = {}
        from pyscf.data import elements
        for atom in set(a[0] for a in mol._atom):
            raw = gto.mole.uncontracted_basis(mol._basis[atom])
            sh = numpy.array([b[0] for b in raw])
            ex = numpy.array([b[-1][0] for b in raw])
            ints[atom] = quiet(lambda: _x2camf_mod.libx2camf.atm_integrals(
                0, elements.charge(gto.mole._std_symbol(atom)), sh[-1] + 1, len(sh), 0, sh, ex,
                speed_of_light=param.LIGHT_SPEED))
    else:
        quiet(lambda: xo.get_hcore(mol))
        ints = xo.atomic_integrals
    xmol, contr = _decontract_spinor(mol, xo.xuncontract)
    n2c = xmol.nao_2c()
    sl = xmol.aoslice_2c_by_atom()
    gm = lambda i, four: construct_molecular_matrix(X.extract_ith_integral(ints, i), sl, xmol, n2c, four)
    d4 = gm(11, True)
    d2 = gm(12, False)
    m4 = scf.DHF(xmol)
    m4.verbose = 0
    h4, s4 = m4.get_hcore(), m4.get_ovlp()
    mfnr = scf.RHF(mol)
    dmnr = mfnr.get_init_guess()
    opt = mfnr.init_direct_scf(mol)
    t0 = time.perf_counter()
    scf.hf.get_jk(mol, dmnr, vhfopt=opt, with_k=False)
    t_nrj = time.perf_counter() - t0
    t0 = time.perf_counter()
    scf.hf.get_jk(mol, dmnr, vhfopt=opt)
    t_nr = time.perf_counter() - t0
    t0 = time.perf_counter()
    X.x2c1e_hfw0_4cmat(h4, s4, mol=xmol)
    t_x2c = time.perf_counter() - t0
    if large:
        ca, cb = xmol.sph2spinor_coeff()
        p2 = sum((c.conj() @ d2 @ c.T) for c in (ca, cb)).real
        vjs = scf.hf.get_jk(xmol, p2, with_k=False)[0]
        vj2 = ca.conj().T @ vjs @ ca + cb.conj().T @ vjs @ cb
        for method in ('fit',):
            j4, j2, tt, ng = j_fast(xmol, d4, d2, method)
            print('%-7s %5d %4d | %6.2f %6.2f %6.2f | %-4s %4dk %6.2f %6.2f %6.2f %6.2f %7.1f | %9.1e % 9.1e' % (
                name, n2c, mol.natm, t_nrj, t_nr, t_x2c, method, ng // 1000, tt[0], tt[1], tt[2], tt.sum(),
                tt.sum() / t_nr, abs(j2 - vj2).max(), numpy.einsum('ij,ji', d2, j2 - vj2).real), flush=True)
        continue
    t0 = time.perf_counter()
    vj4 = m4.get_jk(xmol, d4, with_k=False)[0]
    vj2 = spinor_hf.SCF(xmol).get_jk(xmol, d2, with_k=False)[0]
    t_ex = time.perf_counter() - t0

    def hcore(j4, j2):
        x, _, r, _ = X.x2c1e_hfw0_4cmat(h4 + j4, s4, mol=xmol)
        return contr.conj().T @ (X.to_2c(x, r, h4 + j4) - j2) @ contr
    h_ex = hcore(vj4, vj2)

    def energy(j4, j2):
        hc = hcore(j4, j2)
        m = spinor_hf.SCF(mol)
        m.get_hcore = lambda *a: hc.copy()
        m.conv_tol = 1e-11
        m.verbose = 0
        return quiet(m.kernel)
    do_e = n2c <= 400
    e_ex = energy(vj4, vj2) if do_e else None
    for method in ('erf', 'fit'):
        j4, j2, tt, ng = j_fast(xmol, d4, d2, method)
        de = '% 9.1e' % (energy(j4, j2) - e_ex) if do_e else '%9s' % '-'
        print('%-7s %5d | %6.2f %8.2f %6.2f | %-4s %4dk %6.2f %6.2f %6.2f %6.2f %6.1f | %9.1e %9.1e %9.1e %s' % (
            name, n2c, t_nr, t_ex, t_x2c, method, ng // 1000, tt[0], tt[1], tt[2], tt.sum(), tt.sum() / t_nr,
            abs(j4 - vj4)[:n2c, :n2c].max(), abs(j4 - vj4)[n2c:, n2c:].max(), abs(hcore(j4, j2) - h_ex).max(), de),
            flush=True)

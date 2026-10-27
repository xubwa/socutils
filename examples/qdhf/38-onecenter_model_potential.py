#!/usr/bin/env python

'''
One-center model potential: x2cmp rebuilt from one-center quantities only.

x2cmp (somf/x2cmp.py, amf_type 'x2cmp') decouples with
    F4 = h4 + (J - K)[sum_K D4_K]      (molecular 2e integrals)
and subtracts the 2c counterpart (J - K)[sum_K D2_K].  Here
  * J: the atomic densities are spherical, so rho_K(r) (4c: L + S; 2c) is
    Coulomb-fitted (charge-conserving) to normalized s Gaussians; J is then
    sum_k q_k erf(sqrt(zeta_k) r)/r -- the SAP form, but from our own atomic
    density -- and needs only 1e (rinv_zeta) integrals, all blocks;
  * K: one-center, block-diagonal (atom-only 2e integrals).
Variants:  'Jfit+K1c' (the scheme), 'Jx+K1c' (exact molecular J, one-center K),
'Jfit+K1cP' (one-center K in projector form S_.K S_K^-1 K_K S_K^-1 S_K.),
'Jnum+K1c(P)': J numerically exact instead of fitted -- the spherical atomic
density's Hartree potential from the radial Poisson equation (cumulative
Simpson on a fine log grid), N erf(r)/r long range analytic (rinv_zeta), the
bounded short-range remainder on the molecular Becke grid (LL with spinor AO
values, SS with sigma.p AO values);
'Jfit+Kx' (fit check: molecular K).  Reported against x2cmp itself:
E - E(x2cmp), max |d eps_occ|, lowest h_core eigenvalue.

Usage:  python 38-onecenter_model_potential.py kmax geom basis [fit_ratio=1.2] [jgrid=150,590; 0,0 = off]
'''

import os
import sys
import time
import contextlib
import numpy
import scipy.linalg
import scipy.special
from functools import reduce
from pyscf import gto, scf, lib
lib.einsum = lambda s, *o, **k: numpy.einsum(s, *o, optimize=True)
# pyscf.lib.diis catches numpy.linalg.linalg.LinAlgError (gone in numpy 2.x)
numpy.linalg.linalg = numpy.linalg
import socutils
from pyscf.lib import param
from pyscf.x2c.x2c import _decontract_spinor
from socutils.scf import spinor_hf
from socutils.somf import x2cmp as X
from x2camf.x2camf import construct_molecular_matrix

kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 8
geom = sys.argv[2] if len(sys.argv) > 2 else 'H 0 0 0; Cl 0 0 1.275'
base = sys.argv[3] if len(sys.argv) > 3 else 'unc-cc-pvdz'
fit_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 1.2
# Becke grid (radial, angular) for the short-range part of the numerical J.  HCl k=0,
# E error vs exact molecular J: (75,194) 2e-7, (100,302) 3e-9, (150,590) 1e-11, ~1 s
jgrid = tuple(int(x) for x in sys.argv[5].split(',')) if len(sys.argv) > 5 else (150, 590)
os.environ['X2CAMF_TIMING'] = '0'
elems = sorted(set(a.split()[0] for a in geom.split(';')))
base = dict(x.split(':') for x in base.split(',')) if ':' in base else {el: base for el in elems}


def make_basis(k):
    bas = {}
    for el in elems:
        name = base[el]
        b = gto.basis.load(name[4:] if name.startswith('unc-') else name, el)
        raw = gto.mole.uncontracted_basis(b)
        if name.startswith('unc-'):
            b = raw
        extra = []
        for l in (0, 1):
            emin = min(x[-1][0] for x in raw if x[0] == l)
            extra += [[l, [emin / 2.5**(i+1), 1.]] for i in range(k)]
        bas[el] = sorted(b + extra, key=lambda sh: sh[0])
    return bas


def quiet(f):
    with open(os.devnull, 'w') as fn, contextlib.redirect_stdout(fn):
        return f()


def projector_form(a_blocks, s_full, slices):
    out = numpy.zeros_like(s_full)
    for a, idx in zip(a_blocks, slices):
        sk = s_full[numpy.ix_(idx, idx)]
        m = scipy.linalg.solve(sk, scipy.linalg.solve(sk, a).conj().T).conj().T
        sxk = s_full[:, idx]
        out += sxk @ m @ sxk.conj().T
    return out

def atom_mol(xmol, ia):
    symb = xmol.atom_symbol(ia)
    return gto.Mole(atom=[(symb, (0., 0., 0.))], basis={symb: xmol._basis[symb]}, unit='bohr', verbose=0, spin=xmol.atom_charge(ia) % 2).build()

def radial_density(amol, d, comp, nrad=3000, rmin=1e-7, rmax=2e3, nang=6):
    '''spherically averaged density 4 pi r^2 rho(r) on a log grid from a 2c (comp='L')
    or small-component (comp='S', pyscf DHF convention sigma.p chi / 2c) density
    block.  The atomic densities are spherical, so a 6-point Lebedev average
    is exact (checked against 26 points); contraction via zgemm.'''
    r = numpy.exp(numpy.linspace(numpy.log(rmin), numpy.log(rmax), nrad))
    from pyscf.dft.LebedevGrid import MakeAngularGrid
    ang = MakeAngularGrid(nang)
    rho = numpy.zeros(nrad)
    c = param.LIGHT_SPEED
    n = d.shape[0]
    for xyz, w in zip(ang[:, :3], ang[:, 3]):
        coords = r[:, None] * xyz[None, :]
        if comp == 'L':
            ao = amol.eval_gto('GTOval_spinor', coords)
        else:
            ao = amol.eval_gto('GTOval_sp_spinor', coords) * (.5 / c)
        ao = ao.reshape(-1, n)
        rho += w * ((ao.conj() @ d) * ao).real.sum(axis=1).reshape(2, -1).sum(axis=0)
    return r, 4 * numpy.pi * rho * r**2   # Lebedev weights sum to 1

def coulomb_fit(r, rad, zetas):
    '''Gaussian charges q_k (normalized s Gaussians of exponent zeta_k) that
    best reproduce the spherical density in the Coulomb metric with
    sum q_k = N.  rad = 4 pi r^2 rho(r) on the log grid r.'''
    lnr = numpy.log(r)
    integ = lambda f: numpy.trapezoid(f * r, lnr)
    n = integ(rad)
    a = numpy.asarray(zetas)
    # (g_a|g_b) = 2/sqrt(pi) sqrt(ab/(a+b));  (g_a|rho) = int rad(r) erf(sqrt(a) r)/r dr
    m = 2 / numpy.sqrt(numpy.pi) * numpy.sqrt(numpy.outer(a, a) / numpy.add.outer(a, a))
    b = numpy.array([integ(rad * scipy.special.erf(numpy.sqrt(x) * r) / r) for x in a])
    nk = len(a)
    kkt = numpy.zeros((nk + 1, nk + 1))
    kkt[:nk, :nk] = m
    kkt[:nk, nk] = kkt[nk, :nk] = 1
    sol = numpy.linalg.lstsq(kkt, numpy.append(b, n), rcond=1e-14)[0]
    q = sol[:nk]
    # Coulomb self-energy error  (rho-fit|rho-fit) = (rho|rho) - 2 q.b + q.m.q
    return q, n

def fit_exponents(amol, ratio=2.0):
    ex = numpy.concatenate([amol.bas_exp(i) for i in range(amol.nbas)])
    lo, hi = 2 * ex.min(), 2 * ex.max() * 4
    nz = int(numpy.ceil(numpy.log(hi / lo) / numpy.log(ratio))) + 1
    return lo * ratio**numpy.arange(nz)

def jfit_matrices(xmol, ia, q, zetas):
    c = param.LIGHT_SPEED
    v = 0
    w = 0
    with xmol.with_rinv_origin(xmol.atom_coord(ia)):
        for qk, z in zip(q, zetas):
            with xmol.with_rinv_zeta(z):
                v = v + qk * xmol.intor('int1e_rinv_spinor')
                w = w + qk * xmol.intor('int1e_sprinvsp_spinor')
    return v, w * (.25 / c**2)


from scipy.integrate import cumulative_simpson
from pyscf.dft import gen_grid
import scipy.interpolate


def radial_hartree(r, rad):
    '''V(r) of the spherical density rad = 4 pi r^2 rho on the log grid r:
    V(r) = 1/r int_0^r rad dr' + int_r^inf rad/r' dr'  (cumulative trapezoid in ln r)'''
    lnr = numpy.log(r)
    c1 = cumulative_simpson(rad * r, x=lnr, initial=0.)     # int_0^r rad dr'
    c2 = cumulative_simpson(rad, x=lnr, initial=0.)         # int_0^r rad/r' dr'
    return c1 / r + (c2[-1] - c2), c1[-1]

def vsr_spline(am, dl, ds, zeta0=1.0, nlog=20000):
    '''radial Hartree potential of the spherical atomic density (dl: LL block,
    ds: SS block or None), split as N erf(sqrt(zeta0) r)/r + V_sr(r); returns
    (N, spline of V_sr in ln r, r grid)'''
    r, rad = radial_density(am, dl, 'L', nrad=nlog)
    if ds is not None:
        rad = rad + radial_density(am, ds, 'S', nrad=nlog)[1]
    v, ntot = radial_hartree(r, rad)
    vsr = v - ntot * scipy.special.erf(numpy.sqrt(zeta0) * r) / r
    return ntot, scipy.interpolate.CubicSpline(numpy.log(r), vsr), r


def jnum(xmol, atoms, zeta0=1.0, nrad=200, nang=1202, blk=20000):
    '''numerically exact J of superposed spherical atomic densities as 4c
    (pyscf DHF convention) and 2c matrices.  atoms: per atom (ia, N4, spl4, N2,
    spl2, r).  Long range N erf/r analytic (rinv_zeta).  The short-range rest,
    summed over atoms, is spin-independent, so it is integrated on one
    molecular Becke grid with real scalar AOs and their gradients only:
      LL: A = <phi|V|phi>;  SS: (s.p chi)^+ V (s.p chi) = sum_ij A_ij (x) s_i s_j,
      A_ij = <d_i phi|V|d_j phi>, s_i s_j = delta_ij + i eps_ijk s_k,
    then carried to spinors with sph2spinor_coeff.'''
    c = param.LIGHT_SPEED
    n2c = xmol.nao_2c()
    nao = xmol.nao
    ll4 = numpy.zeros((n2c, n2c), complex)
    ss4 = numpy.zeros_like(ll4)
    ll2 = numpy.zeros_like(ll4)
    for ia, n4, spl4, n2, spl2, r in atoms:
        with xmol.with_rinv_origin(xmol.atom_coord(ia)), xmol.with_rinv_zeta(zeta0):
            v = xmol.intor('int1e_rinv_spinor')
            ll4 += n4 * v
            ll2 += n2 * v
            ss4 += n4 * xmol.intor('int1e_sprinvsp_spinor') * (.25 / c**2)
    g = gen_grid.Grids(xmol)
    g.atom_grid = (nrad, nang)
    g.prune = None
    g.build(with_non0tab=False)
    v4 = numpy.zeros(len(g.weights))
    v2 = numpy.zeros(len(g.weights))
    for ia, n4, spl4, n2, spl2, r in atoms:
        d = numpy.linalg.norm(g.coords - xmol.atom_coord(ia), axis=1)
        inside = d < r[-1]
        x = numpy.log(numpy.maximum(d[inside], r[0]))
        v4[inside] += spl4(x)
        v2[inside] += spl2(x)
    w4, w2 = g.weights * v4, g.weights * v2
    a4 = numpy.zeros((nao, nao))
    a2 = numpy.zeros((nao, nao))
    aij = numpy.zeros((3, 3, nao, nao))
    for p0 in range(0, len(w4), blk):
        sl = slice(p0, p0 + blk)
        ao = xmol.eval_gto('GTOval_sph_deriv1', g.coords[sl])      # (4, g, nao)
        a4 += (ao[0] * w4[sl, None]).T @ ao[0]
        a2 += (ao[0] * w2[sl, None]).T @ ao[0]
        for i in range(3):
            wi = ao[i + 1] * w4[sl, None]
            for j in range(3):
                aij[i, j] += wi.T @ ao[j + 1]
    ca, cb = xmol.sph2spinor_coeff()
    sc = lambda m: ca.conj().T @ m @ ca + cb.conj().T @ m @ cb
    ll4 += sc(a4)
    ll2 += sc(a2)
    w0 = aij[0, 0] + aij[1, 1] + aij[2, 2]
    wx = aij[1, 2] - aij[2, 1]
    wy = aij[2, 0] - aij[0, 2]
    wz = aij[0, 1] - aij[1, 0]
    # W0 (x) 1 + i (Wx (x) s_x + Wy (x) s_y + Wz (x) s_z) in the (alpha, beta) spin-orbital basis
    maa = w0 + 1j * wz
    mbb = w0 - 1j * wz
    mab = 1j * wx + wy          # i (Wx - i Wy)
    mba = 1j * wx - wy          # i (Wx + i Wy)
    ss = (ca.conj().T @ maa @ ca + ca.conj().T @ mab @ cb + cb.conj().T @ mba @ ca + cb.conj().T @ mbb @ cb)
    ss4 += ss * (.25 / c**2)
    return scipy.linalg.block_diag(ll4, ss4), ll2


def one_k(k):
    mol = gto.M(atom=geom, basis=make_basis(k), verbose=0)
    mf = spinor_hf.SCF(mol)
    mf.x2cmp(flavor='x2cmp', with_gaunt=False, with_breit=False, with_pcc=False)
    xo = mf.with_x2c
    h_mp = quiet(lambda: xo.get_hcore(mol))
    ints = xo.atomic_integrals
    xmol, contr = _decontract_spinor(mol, xo.xuncontract)
    n2c = xmol.nao_2c()
    sl = xmol.aoslice_2c_by_atom()
    gm = lambda i, four: construct_molecular_matrix(X.extract_ith_integral(ints, i), sl, xmol, n2c, four)
    idx2 = [numpy.arange(p0, p1) for p0, p1 in sl[:, 2:4]]
    idx4 = [numpy.concatenate([i, i + n2c]) for i in idx2]
    m4 = scf.DHF(xmol)
    m4.verbose = 0
    h4, s4 = m4.get_hcore(), m4.get_ovlp()
    s2 = s4[:n2c, :n2c]
    smin = numpy.linalg.eigvalsh(s2).min()
    d4 = gm(11, True)
    d2 = gm(12, False)
    vj4, vk4 = m4.get_jk(xmol, d4)
    vj2, vk2 = spinor_hf.SCF(xmol).get_jk(xmol, d2)
    k4b, k2b, j4f, j2f, j4n, j2n = [numpy.zeros_like(a) for a in (vk4, vk2, vj4, vj2, vj4, vj2)]
    k4blocks, k2blocks = [], []
    atoms_j = []
    errn = 0.
    for ia, (i2, i4) in enumerate(zip(idx2, idx4)):
        am = atom_mol(xmol, ia)
        nk = len(i2)
        b4 = d4[numpy.ix_(i4, i4)]
        b2 = d2[numpy.ix_(i2, i2)]
        ma4 = scf.DHF(am)
        ma4.verbose = 0
        k4blocks.append(ma4.get_jk(am, b4)[1])
        k2blocks.append(spinor_hf.SCF(am).get_jk(am, b2)[1])
        k4b[numpy.ix_(i4, i4)] = k4blocks[-1]
        k2b[numpy.ix_(i2, i2)] = k2blocks[-1]
        r, rad_l = radial_density(am, b4[:nk, :nk], 'L', nrad=6000)
        _, rad_s = radial_density(am, b4[nk:, nk:], 'S', nrad=6000)
        _, rad2 = radial_density(am, b2, 'L', nrad=6000)
        z = fit_exponents(am, fit_ratio)
        q4, n4 = coulomb_fit(r, rad_l + rad_s, z)
        q2, n2 = coulomb_fit(r, rad2, z)
        errn = max(errn, abs(n4 - xmol.atom_charge(ia)), abs(n2 - xmol.atom_charge(ia)))
        v, w = jfit_matrices(xmol, ia, q4, z)
        j4f += scipy.linalg.block_diag(v, w)
        j2f += jfit_matrices(xmol, ia, q2, z)[0]
        if jgrid[0]:
            n4n, spl4, rr = vsr_spline(am, b4[:nk, :nk], b4[nk:, nk:])
            n2n, spl2, _ = vsr_spline(am, b2, None)
            atoms_j.append((ia, n4n, spl4, n2n, spl2, rr))
    if jgrid[0]:
        j4n, j2n = jnum(xmol, atoms_j, 1.0, *jgrid)
    errj = (abs(j4n - vj4).max(), abs(j2n - vj2).max()) if jgrid[0] else (numpy.nan, numpy.nan)
    k4p = projector_form(k4blocks, s4, idx4)
    k2p = projector_form(k2blocks, s2, idx2)

    cc = lambda h: reduce(numpy.dot, (contr.conj().T, h, contr))

    def build(veff4, veff2):
        x, _, r, _ = X.x2c1e_hfw0_4cmat(h4 + veff4, s4, mol=xmol)
        return cc(X.to_2c(x, r, h4 + veff4) - veff2)
    hs = {'x2cmp': h_mp, 'Jfit+K1c': build(j4f - k4b, j2f - k2b), 'Jx+K1c': build(vj4 - k4b, vj2 - k2b),
          'Jfit+K1cP': build(j4f - k4p, j2f - k2p), 'Jfit+Kx': build(j4f - vk4, j2f - vk2)}
    if jgrid[0]:
        hs['Jnum+K1c'] = build(j4n - k4b, j2n - k2b)
        hs['Jnum+K1cP'] = build(j4n - k4p, j2n - k2p)
    sc = mol.intor('int1e_ovlp_spinor')
    res = {}
    for key, hc in hs.items():
        m = spinor_hf.SCF(mol)
        m.get_hcore = lambda *a: hc.copy()
        m.conv_tol = 1e-10
        m.max_cycle = 100
        m.verbose = 0
        e = quiet(m.kernel)
        res[key] = (e, m.converged, numpy.sort(m.mo_energy[m.mo_occ > 0])[::2],
                    scipy.linalg.eigh(hc, sc, eigvals_only=True).min())
    return smin, (errn,) + errj, res


keys = ['x2cmp', 'Jfit+K1c', 'Jx+K1c', 'Jfit+K1cP', 'Jfit+Kx'] + (['Jnum+K1c', 'Jnum+K1cP'] if jgrid[0] else [])
print('# %s, %s + k diffuse s,p (ratio 2.5); Dirac-Coulomb; J fit ratio %.2f' % (geom, base, fit_ratio))
rows = []
for k in range(kmax + 1):
    t0 = time.time()
    try:
        rows.append((k,) + one_k(k))
    except Exception as ex:
        print('# k=%d failed: %r' % (k, ex), flush=True)
        break
    k_, smin, errn, res = rows[-1]
    e0 = res['x2cmp'][0]
    print('# k=%d %.0f s Smin %.2e |N_fit - Z| %.1e |Jnum - J| 4c %.1e 2c %.1e  E-E(x2cmp): ' % ((k, time.time() - t0, smin) + errn) +
          ' '.join('%s % .3e%s' % (s, res[s][0] - e0, '' if res[s][1] else '*') for s in keys[1:]), flush=True)

print('\n## E - E(x2cmp) / Eh  (* = not converged);  E(x2cmp) itself in column 3')
print('%2s %9s %18s ' % ('k', 'Smin', 'E(x2cmp)') + ''.join('%-13s' % s for s in keys[1:]))
for k, smin, errn, res in rows:
    e0 = res['x2cmp'][0]
    print('%2d %9.2e %18.10f%s' % (k, smin, e0, ' ' if res['x2cmp'][1] else '*') +
          ''.join('% .5e%s ' % (res[s][0] - e0, ' ' if res[s][1] else '*') for s in keys[1:]))
print('\n## max |eps_occ - eps_occ(x2cmp)| / Eh')
print('%2s ' % 'k' + ''.join('%-13s' % s for s in keys[1:]))
for k, smin, errn, res in rows:
    print('%2d ' % k + ''.join('%12.3e ' % abs(res[s][2] - res['x2cmp'][2]).max() for s in keys[1:]))
print('\n## lowest h_core eigenvalue / Eh')
print('%2s ' % 'k' + ''.join('%-14s' % s for s in keys))
for k, smin, errn, res in rows:
    print('%2d ' % k + ''.join('%13.6f ' % res[s][3] for s in keys))
print('\n## occupied orbital energies (one per Kramers pair), x2cmp and Jfit+K1c / Eh')
for k, smin, errn, res in rows:
    for s in ('x2cmp', 'Jfit+K1c'):
        print('%2d %-9s' % (k, s) + ''.join('%12.6f' % x for x in res[s][2]))

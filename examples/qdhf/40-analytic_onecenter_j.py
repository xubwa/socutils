#!/usr/bin/env python

'''
Exact analytic Coulomb matrix of superposed spherical atomic densities (the J
of the one-center model potential, 38-onecenter_model_potential.py), with
existing libcint integrals only, and its cost against one scalar NR JK build.

 1. exact expansion (per element): the spherically averaged atomic density is
    rho_K(r) = sum c t_{n,zeta}(r),  t = r^2n exp(-zeta r^2), zeta = sums of
    primitive exponents.  LL: rho_L = chi^T P chi in the cart AO basis, the
    angular average of x^p y^q z^s is (p-1)!!(q-1)!!(s-1)!!/(p+q+s+1)!!;
    SS: d_j phi expanded exactly in l+-1 cart shells of the same exponent
    (derivative matrices D_j), rho_S = sum_jk d_j phi^T Q^jk d_k phi.
 2. potential: Laplacian recursion
      V[t_{n+1}] = ((8n+6) z V[t_n] - 2n(2n+1) V[t_{n-1}] - 4 pi t_n) / (4 z^2),
      V[t_0] = (pi/z)^(3/2) erf(sqrt(z) r)/r,
    so V = sum_z E_z erf(sqrt(z) r)/r + sum_{m,z} L_mz r^2m exp(-z r^2).
 3. matrix elements, one contracted s "shell" per atom (all its zetas):
      erf part   : int3c2e     (the erf/r of a unit-charge Gaussian)
      local part : int3c1e, int3c1e_r2/r4/r6_origk
    LL in the spherical AO basis; SS = sum_ij <d_i phi|V|d_j phi> s_i s_j from the
    same integrals over the derivative (cart) shells.  4c and 2c densities share
    every integral call (two k shells per atom).

Checked against the exact molecular J (pyscf DHF get_jk / spinor get_jk):
1e-12 in LL and 2c, ~1e-8 absolute in SS.

Usage:  python 40-analytic_onecenter_j.py [small|large]
'''

import os
import sys
import time
import contextlib
import numpy
import scipy.linalg
from pyscf import gto, scf, lib
lib.einsum = lambda s, *o, **k: numpy.einsum(s, *o, optimize=True)
numpy.linalg.linalg = numpy.linalg
import socutils
from pyscf.lib import param
from pyscf.data import elements
from pyscf.df import incore
from pyscf.gto.mole import (ATM_SLOTS, BAS_SLOTS, PTR_COORD, ATOM_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF,
                            PTR_ENV_START, gaussian_int)
from pyscf.x2c.x2c import _decontract_spinor
from socutils.scf import spinor_hf
from socutils.somf import x2cmp as X
import x2camf
from x2camf.x2camf import construct_molecular_matrix

os.environ['X2CAMF_TIMING'] = '0'


def quiet(f):
    with open(os.devnull, 'w') as fn, contextlib.redirect_stdout(fn):
        return f()




def vt_coeffs(n, zeta):
    '''V[t_n](r) = a erf(sqrt(zeta) r)/r + sum_m b[m] r^2m exp(-zeta r^2) for
    t_n = r^2n exp(-zeta r^2), from
    V[t_{n+1}] = ((8n+6) zeta V[t_n] - 2n(2n+1) V[t_{n-1}] - 4 pi t_n) / (4 zeta^2)'''
    vs = [((numpy.pi / zeta)**1.5, numpy.zeros(0))]
    for k in range(n):
        a0, b0 = vs[k]
        a1, b1 = vs[k - 1] if k >= 1 else (0., numpy.zeros(0))
        b = numpy.zeros(k + 1)
        b[:len(b0)] += (8 * k + 6) * zeta * b0
        b[:len(b1)] -= 2 * k * (2 * k + 1) * b1
        b[k] -= 4 * numpy.pi
        vs.append((((8 * k + 6) * zeta * a0 - 2 * k * (2 * k + 1) * a1) / (4 * zeta**2), b / (4 * zeta**2)))
    return vs[n]


def vt_eval(n, zeta, r):
    a, b = vt_coeffs(n, zeta)
    return a * scipy.special.erf(numpy.sqrt(zeta) * r) / r + sum(bm * r**(2 * m) * numpy.exp(-zeta * r**2) for m, bm in enumerate(b))


_PAULI = numpy.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


def _dfact(n):
    return 1. if n <= 0 else float(numpy.prod(numpy.arange(n, 0, -2)))


def ang_avg(p, q, s):
    '''<x^p y^q z^s / r^(p+q+s)> over the unit sphere'''
    if p % 2 or q % 2 or s % 2:
        return 0.
    return _dfact(p - 1) * _dfact(q - 1) * _dfact(s - 1) / _dfact(p + q + s + 1)


def cart_powers(l):
    return [(a, b, l - a - b) for a in range(l, -1, -1) for b in range(l - a, -1, -1)]


def shell_info(mol):
    '''per shell: (l, alpha, K) with cart AO = K x^a y^b z^c exp(-alpha r^2) (uncontracted mol)'''
    out = []
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        a = mol.bas_exp(ib)
        assert len(a) == 1 and mol.bas_nctr(ib) == 1
        a = a[0]
        p = numpy.array([[0.31, 0.47, 0.23]]) / numpy.sqrt(a) + mol.bas_coord(ib)
        v = mol.eval_gto('GTOval_cart', p, shls_slice=(ib, ib + 1))[0, 0]       # x^l component
        x = p[0] - mol.bas_coord(ib)
        out.append((l, a, v / (x[0]**l * numpy.exp(-a * x @ x))))
    return out


def deriv_mol(mol):
    '''cart mol with, per shell (l, alpha), shells (l+1, alpha) and (l-1, alpha); and
    D[i] (ncart_d x ncart) with d_i chi_mu = sum_P D[i][P, mu] chi'_P (all cart)'''
    info = shell_info(mol)
    bas = []
    for l, a, k in info:
        bas.append((l + 1, a))
        if l > 0:
            bas.append((l - 1, a))
    # build per atom
    basis = {}
    atoms = []
    for ia in range(mol.natm):
        label = 'X%d' % ia
        sh = [[b[0], [b[1], 1.]] for ib in range(mol.nbas) if mol.bas_atom(ib) == ia
              for b in ([(info[ib][0] + 1, info[ib][1])] + ([(info[ib][0] - 1, info[ib][1])] if info[ib][0] > 0 else []))]
        basis[label] = sh
        atoms.append((label, mol.atom_coord(ia)))
    dm = gto.Mole(atom=atoms, basis=basis, unit='bohr', cart=True, verbose=0, charge=0, spin=None)
    dm.nelectron = 0
    dm.build(0, 0)
    dinfo = shell_info(dm)
    loc = mol.ao_loc_nr(cart=True)
    dloc = dm.ao_loc_nr(cart=True)
    D = numpy.zeros((3, dloc[-1], loc[-1]))
    # pyscf reorders shells by angular momentum: look the derivative shells up by (atom, l, alpha)
    where = {(dm.bas_atom(jb), dinfo[jb][0], dinfo[jb][1]): jb for jb in range(dm.nbas)}
    for ib, (l, a, k) in enumerate(info):
        ia = mol.bas_atom(ib)
        up = where[(ia, l + 1, a)]
        dn = where[(ia, l - 1, a)] if l > 0 else None
        kup = dinfo[up][2]
        kdn = dinfo[dn][2] if dn is not None else None
        pw, pwu = cart_powers(l), cart_powers(l + 1)
        pwd = cart_powers(l - 1) if l > 0 else []
        for m, (px, py, pz) in enumerate(pw):
            e = numpy.array([px, py, pz])
            for i in range(3):
                eu = e.copy(); eu[i] += 1
                D[i, dloc[up] + pwu.index(tuple(eu)), loc[ib] + m] += -2 * a * k / kup
                if e[i] > 0:
                    ed = e.copy(); ed[i] -= 1
                    D[i, dloc[dn] + pwd.index(tuple(ed)), loc[ib] + m] += e[i] * k / kdn
    return dm, D, dinfo


def expansion(mol, pc, dinfo_mol=None):
    '''exact spherical expansion sum c t_{n,zeta} of rho(r) = chi^T pc chi (one atom-centred
    cart basis mol, pc in its cart AO basis): dict (n, zeta) -> c'''
    info = shell_info(mol) if dinfo_mol is None else dinfo_mol
    loc = mol.ao_loc_nr(cart=True)
    out = {}
    for ib, (l1, a1, k1) in enumerate(info):
        p1 = cart_powers(l1)
        for jb, (l2, a2, k2) in enumerate(info):
            if (l1 + l2) % 2:
                continue
            p2 = cart_powers(l2)
            blk = pc[loc[ib]:loc[ib + 1], loc[jb]:loc[jb + 1]]
            c = 0.
            for m1, e1 in enumerate(p1):
                for m2, e2 in enumerate(p2):
                    if blk[m1, m2] != 0:
                        c += blk[m1, m2] * ang_avg(e1[0] + e2[0], e1[1] + e2[1], e1[2] + e2[2])
            if c != 0:
                key = ((l1 + l2) // 2, round(a1 + a2, 12))
                out[key] = out.get(key, 0.) + c * k1 * k2
    return out


def atom_expansions(am, dl, ds=None):
    '''exact expansions of the LL (and SS, pyscf DHF convention / 4c^2) spherical densities
    of a one-atom spinor density; am is the atom-only (uncontracted) mol'''
    cs = am.sph2spinor_coeff()
    c2s = am.cart2sph_coeff()
    p = sum((cs[x].conj() @ dl @ cs[x].T) for x in (0, 1)).real
    exp_l = expansion(am, c2s @ p @ c2s.T)
    if ds is None:
        return exp_l, None
    c = param.LIGHT_SPEED
    pyy = [[cs[y].conj() @ ds @ cs[yp].T for yp in (0, 1)] for y in (0, 1)]
    dm, D, dinfo = deriv_mol(am)
    ps = 0
    for j in range(3):
        for k in range(3):
            sjsk = _PAULI[j] @ _PAULI[k]
            q = sum(sjsk[y, yp] * pyy[y][yp] for y in (0, 1) for yp in (0, 1)).real * (.25 / c**2)
            ps = ps + D[j] @ c2s @ q @ c2s.T @ D[k].T
    return exp_l, expansion(dm, ps, dinfo)


_LOC_INTOR = {0: 'int3c1e', 1: 'int3c1e_r2_origk', 2: 'int3c1e_r4_origk', 3: 'int3c1e_r6_origk'}


def kmol_s(coords, prims, cart=False):
    '''fake mol of contracted s shells: shell i at coords[i] with primitives prims[i] =
    (exponents, raw coefficients); the coefficients multiply exp(-zeta r^2) exactly
    (libcint's 1/(2 sqrt(pi)) s factor compensated)'''
    nb = len(prims)
    atm = numpy.zeros((nb, ATM_SLOTS), dtype=numpy.int32)
    bas = numpy.zeros((nb, BAS_SLOTS), dtype=numpy.int32)
    env = [numpy.zeros(PTR_ENV_START)]
    ptr = PTR_ENV_START
    for i, (xyz, (ez, cf)) in enumerate(zip(coords, prims)):
        atm[i, PTR_COORD] = ptr
        env.append(numpy.asarray(xyz, float))
        ptr += 3
        bas[i, ATOM_OF] = i
        bas[i, NPRIM_OF] = len(ez)
        bas[i, NCTR_OF] = 1
        bas[i, PTR_EXP] = ptr
        env.append(numpy.asarray(ez, float))
        ptr += len(ez)
        bas[i, PTR_COEFF] = ptr
        env.append(numpy.asarray(cf, float) * 2 * numpy.sqrt(numpy.pi))
        ptr += len(ez)
    m = gto.Mole()
    m._atm, m._bas, m._env = atm, bas, numpy.hstack(env)
    m.cart = cart
    m._built = True
    return m


def potential_terms(expn):
    '''sum c V[t_{n,zeta}] -> erf part {zeta: E} and local parts {m: {zeta: L}}'''
    erf, loc = {}, {}
    for (n, z), c in expn.items():
        a, b = vt_coeffs(n, z)
        erf[z] = erf.get(z, 0.) + c * a      # erf(sqrt(z) r)/r = potential of the unit-charge Gaussian
        for m, bm in enumerate(b):
            loc.setdefault(m, {})
            loc[m][z] = loc[m].get(z, 0.) + c * bm
    return erf, loc


def j_analytic_exact(xmol, atom_expn, timing=None):
    '''exact J of superposed spherical atomic densities.  atom_expn[ia] = (expansion of
    the 4c density L+S, expansion of the 2c density), each {(n, zeta): c}.
    Returns J4 (pyscf DHF 4c convention) and J2 (spinor).'''
    t0 = time.perf_counter()
    c = param.LIGHT_SPEED
    natm = xmol.natm
    terms = [(potential_terms(e4), potential_terms(e2)) for e4, e2 in atom_expn]
    coords = [xmol.atom_coord(ia) for ia in range(natm)]
    # erf part: per atom one contracted unit-charge-Gaussian shell for 4c and one for 2c
    prims = []
    for (erf4, _), (erf2, _) in terms:
        for erf in (erf4, erf2):
            z = numpy.array(sorted(erf))
            prims.append((z, numpy.array([erf[x] for x in z]) / (2 * numpy.sqrt(numpy.pi) * gaussian_int(2, z)) / (2 * numpy.sqrt(numpy.pi))))
    kco = [coords[i // 2] for i in range(2 * natm)]
    kerf = kmol_s(kco, prims)
    kerf_c = kmol_s(kco, prims, cart=True)
    # (mu nu|k) is symmetric in mu, nu: lower triangle only (aosym s2ij), unpacked after the k sum
    tri = lambda v, sel: lib.unpack_tril(v[:, sel].sum(axis=1))
    ev, od = numpy.arange(0, 2 * natm, 2), numpy.arange(1, 2 * natm, 2)
    v = incore.aux_e2(xmol, kerf, intor='int3c2e_sph', aosym='s2ij')
    ll4 = tri(v, ev)
    ll2 = tri(v, od)
    t1 = time.perf_counter()
    mmax = max(max(list(t[0][1].keys()) + [-1]) for t in terms)
    kloc = {}
    for m in range(mmax + 1):
        prims = []
        for (_, loc4), (_, loc2) in terms:
            for loc in (loc4, loc2):
                d = loc.get(m, {})
                z = numpy.array(sorted(d)) if d else numpy.array([1.])
                prims.append((z, numpy.array([d[x] for x in z]) if d else numpy.zeros(1)))
        kloc[m] = (kmol_s(kco, prims), kmol_s(kco, prims, cart=True))
        v = incore.aux_e2(xmol, kloc[m][0], intor=_LOC_INTOR[m] + '_sph', aosym='s2ij')
        ll4 += tri(v, ev)
        ll2 += tri(v, od)
    t2 = time.perf_counter()
    # SS: sum_ij <d_i phi|V|d_j phi> s_i s_j via the l+-1 derivative shells (cart), 4c shells only
    dm, D, _ = deriv_mol(xmol)
    c2s = xmol.cart2sph_coeff()
    Dc = [D[i] @ c2s for i in range(3)]
    tss = tri(incore.aux_e2(dm, kerf_c, intor='int3c2e_cart', aosym='s2ij'), ev)
    for m, km in kloc.items():
        tss += tri(incore.aux_e2(dm, km[1], intor=_LOC_INTOR[m] + '_cart', aosym='s2ij'), ev)
    a = [[Dc[i].T @ tss @ Dc[j] for j in range(3)] for i in range(3)]
    t3 = time.perf_counter()
    w0 = a[0][0] + a[1][1] + a[2][2]
    wx, wy, wz = a[1][2] - a[2][1], a[2][0] - a[0][2], a[0][1] - a[1][0]
    ca, cb = xmol.sph2spinor_coeff()
    sc = lambda mm: ca.conj().T @ mm @ ca + cb.conj().T @ mm @ cb
    ss = (ca.conj().T @ (w0 + 1j * wz) @ ca + ca.conj().T @ (1j * wx + wy) @ cb
          + cb.conj().T @ (1j * wx - wy) @ ca + cb.conj().T @ (w0 - 1j * wz) @ cb)
    if timing is not None:
        timing.extend([t1 - t0, t2 - t1, t3 - t2, time.perf_counter() - t3])
    return scipy.linalg.block_diag(sc(ll4), ss * (.25 / c**2)), sc(ll2)


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


def atom_mol(xmol, ia):
    symb = xmol.atom_symbol(ia)
    return gto.Mole(atom=[(symb, (0., 0., 0.))], basis={symb: xmol._basis[symb]}, unit='bohr',
                    verbose=0, spin=xmol.atom_charge(ia) % 2).build()


def element_expansions(xmol, d4, d2):
    '''per atom (4c L+S expansion, 2c expansion), computed once per element'''
    n2c = xmol.nao_2c()
    sl = xmol.aoslice_2c_by_atom()
    cache, out = {}, []
    for ia in range(xmol.natm):
        s = xmol.atom_symbol(ia)
        if s not in cache:
            i2 = numpy.arange(*sl[ia, 2:4])
            i4 = numpy.concatenate([i2, i2 + n2c])
            nk = len(i2)
            b4 = d4[numpy.ix_(i4, i4)]
            am = atom_mol(xmol, ia)
            el, es = atom_expansions(am, b4[:nk, :nk], b4[nk:, nk:])
            e4 = dict(el)
            for k, v in es.items():
                e4[k] = e4.get(k, 0.) + v
            cache[s] = (e4, atom_expansions(am, d2[numpy.ix_(i2, i2)])[0])
        out.append(cache[s])
    return out


def atomic_densities(mol):
    '''atomic x2camf (DC) densities of the uncontracted mol, as 4c and 2c molecular block matrices'''
    ints = {}
    for atom in set(a[0] for a in mol._atom):
        raw = gto.mole.uncontracted_basis(mol._basis[atom])
        sh = numpy.array([b[0] for b in raw])
        ex = numpy.array([b[-1][0] for b in raw])
        ints[atom] = quiet(lambda: x2camf.libx2camf.atm_integrals(
            0, elements.charge(gto.mole._std_symbol(atom)), sh[-1] + 1, len(sh), 0, sh, ex,
            speed_of_light=param.LIGHT_SPEED))
    xmol, contr = _decontract_spinor(mol, None)
    n2c = xmol.nao_2c()
    sl = xmol.aoslice_2c_by_atom()
    gm = lambda i, four: construct_molecular_matrix(X.extract_ith_integral(ints, i), sl, xmol, n2c, four)
    return xmol, contr, gm(11, True), gm(12, False)


large = len(sys.argv) > 1 and sys.argv[1] == 'large'
print('# exact analytic one-center J; times in s; NR J(K): one scalar direct J (JK) build, same basis')
print('# err: max |J - J_exact| over LL, SS and 2c blocks (small: exact molecular 4c and 2c J; large: 2c only, exact')
print('#      2c J from one scalar J build)')
print('%-7s %5s %4s | %6s %6s %6s | %6s %6s %6s %6s %6s %6s | %7s %7s | %8s %8s %8s' % (
    'mol', 'n2c', 'natm', 'NR J', 'NR JK', 'X2C', 'expand', 'erf', 'local', 'SS', 'total', 'integr',
    'tot/JK', 'int/JK', 'err LL', 'err SS', 'err 2c'))
for name, geom, base in systems:
    mol = gto.M(atom=geom, basis=base[4:], verbose=0)
    mol = gto.M(atom=geom, basis={e: gto.mole.uncontracted_basis(gto.basis.load(base[4:], e))
                                  for e in set(a[0] for a in mol._atom)}, verbose=0)
    xmol, contr, d4, d2 = atomic_densities(mol)
    n2c = xmol.nao_2c()
    mfnr = scf.RHF(mol)
    dmnr = mfnr.get_init_guess()
    opt = mfnr.init_direct_scf(mol)
    t0 = time.perf_counter()
    scf.hf.get_jk(mol, dmnr, vhfopt=opt, with_k=False)
    t_nrj = time.perf_counter() - t0
    t0 = time.perf_counter()
    scf.hf.get_jk(mol, dmnr, vhfopt=opt)
    t_nr = time.perf_counter() - t0
    m4 = scf.DHF(xmol)
    m4.verbose = 0
    t0 = time.perf_counter()
    X.x2c1e_hfw0_4cmat(m4.get_hcore(), m4.get_ovlp(), mol=xmol)
    t_x2c = time.perf_counter() - t0
    t0 = time.perf_counter()
    ae = element_expansions(xmol, d4, d2)
    t_exp = time.perf_counter() - t0
    tt = []
    j4, j2 = j_analytic_exact(xmol, ae, tt)
    t_int = sum(tt)
    ca, cb = xmol.sph2spinor_coeff()
    p2 = sum((c.conj() @ d2 @ c.T) for c in (ca, cb)).real
    vjs = scf.hf.get_jk(xmol, p2, with_k=False)[0]
    vj2 = ca.conj().T @ vjs @ ca + cb.conj().T @ vjs @ cb
    if not large:
        vj4 = m4.get_jk(xmol, d4, with_k=False)[0]
        e_ll = '%8.1e' % abs(j4 - vj4)[:n2c, :n2c].max()
        e_ss = '%8.1e' % abs(j4 - vj4)[n2c:, n2c:].max()
    else:
        e_ll = e_ss = '%8s' % '-'
    print('%-7s %5d %4d | %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f | %7.2f %7.2f | %s %s %8.1e' % (
        name, n2c, mol.natm, t_nrj, t_nr, t_x2c, t_exp, tt[0], tt[1], tt[2] + tt[3], t_exp + t_int, t_int,
        (t_exp + t_int) / t_nr, t_int / t_nr, e_ll, e_ss, abs(j2 - vj2).max()), flush=True)

#!/usr/bin/env python

'''
Long-range Coulomb correction for the atom-block-diagonal AMF quantities.

For a neutral atom, -Z_K/r + J[rho_K] is short-ranged, but in the
block-diagonal form only -Z_K/r is a true molecular operator: J[rho_K] lives in
the K-K block alone.  Functions of other atoms that reach into atom K (diffuse
functions) then see an unscreened -Z_K/r tail, which near-linearly-dependent
cross-atom combinations amplify by 1/w_min.

Long-range corrected form of the atomic 4c two-electron Fock in the
decoupling (amfX2C):

    G4c = J_mol[sum_K D4c_K]  +  sum_K blockdiag_K( G4c_K - J_mol[D4c_K]_KK )

The first term is the Coulomb potential of the superposed atomic densities,
computed as a true molecular operator (all blocks, pyscf DHF get_jk); the
block-diagonal remainder (exchange, ...) is short-ranged.

Part A: J/K split check -- (J - K)[D_K]_KK from pyscf == atomic fock_4c_2e.
Part B: amfX2C block / projector / long-range-corrected (LR), E - E(X2C-1e),
        lowest h_core eigenvalue, lowest orbital energy.
        SAP variants: 'SAPX2C' = SAP-X2C of Surjuse and Valeev (arXiv:2601.04174,
        cf. qdhf/x2c.py approx='sap1e'): X from V + V_SAP, V_SAP subtracted
        untransformed (a 1e-type scheme, no atomic 2e correction);
        'SAP+K' = amfX2C with the Coulomb part J_mol[sum D_K] replaced by
        V_SAP (fitted local potential) and the rest block-diagonal.
Part C: first-order energies Tr(D_X2C-1e dF) of the pieces of the x2camf
        spin-orbit AMF correction (DC SO, Gaunt, Breit gauge), block and
        projector form, and their |dF|_orth.

Usage:  python 37-amf_long_range_coulomb.py kmax geom basis
'''

import os
import sys
import time
import contextlib
import numpy
import scipy.linalg
from functools import reduce
from pyscf import gto, scf, lib
lib.einsum = lambda s, *o, **k: numpy.einsum(s, *o, optimize=True)
# pyscf.lib.diis catches numpy.linalg.linalg.LinAlgError (gone in numpy 2.x) when the
# DIIS matrix turns singular -- with near-linearly-dependent bases it does
numpy.linalg.linalg = numpy.linalg
import socutils
import x2camf
from pyscf.data import elements
from pyscf.lib import param
from pyscf.x2c.x2c import _decontract_spinor
from socutils.scf import spinor_hf
from socutils.somf import x2cmp as X
from x2camf.x2camf import construct_molecular_matrix

kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 8
geom = sys.argv[2] if len(sys.argv) > 2 else 'H 0 0 0; Cl 0 0 1.275'
base = sys.argv[3] if len(sys.argv) > 3 else 'unc-cc-pvdz'
os.environ['X2CAMF_TIMING'] = '0'
elems = sorted(set(a.split()[0] for a in geom.split(';')))
if ':' in base:
    base = dict(x.split(':') for x in base.split(','))
else:
    base = {el: base for el in elems}


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


def orth_norm(a, s):
    w, u = numpy.linalg.eigh(s)
    uc = u / numpy.sqrt(w)
    return abs(numpy.linalg.eigvalsh(uc.conj().T @ a @ uc)).max()


def atm_ints(mol, flavor):
    out = {}
    for atom in set(a[0] for a in mol._atom):
        z = elements.charge(gto.mole._std_symbol(atom))
        raw = gto.mole.uncontracted_basis(mol._basis[atom])
        sh = numpy.array([b[0] for b in raw])
        ex = numpy.array([b[-1][0] for b in raw])
        out[atom] = quiet(lambda: x2camf.libx2camf.atm_integrals(flavor, z, sh[-1]+1, len(sh), 0, sh, ex,
                                                                 speed_of_light=param.LIGHT_SPEED))
    return out


def sap_4c(xmol, sap_basis='sapgrasplarge'):
    '''4c matrix (pyscf DHF convention: LL V, SS pV.p / 4c^2) of the electronic
    part of the SAP of neutral atoms (Lehtola's Gaussian fit, as in pyscf
    init_guess='sap' and qdhf.x2c.sap_charges): Gaussian charges q_k, sum -Z
    per atom, V(r) = -sum_k q_k erf(sqrt(zeta_k) r) / r.'''
    from pyscf.gto.basis import load
    v = 0
    w = 0
    for ia in range(xmol.natm):
        symb = xmol.atom_pure_symbol(ia)
        b = load(sap_basis, symb)
        b = b[symb] if isinstance(b, dict) else b
        arr = numpy.asarray(b[0][1:], dtype=float)
        q = arr[:, 1] * (-xmol.atom_charge(ia) / arr[:, 1].sum())
        for zeta, qk in zip(arr[:, 0], q):
            with xmol.with_rinv_origin(xmol.atom_coord(ia)), xmol.with_rinv_zeta(zeta):
                v = v - qk * xmol.intor('int1e_rinv_spinor')
                w = w - qk * xmol.intor('int1e_sprinvsp_spinor')
    c = param.LIGHT_SPEED
    return scipy.linalg.block_diag(v, w * (.25 / c**2))


def scf_run(mol, hc):
    mf = spinor_hf.SCF(mol)
    mf.get_hcore = lambda *a: hc.copy()
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.verbose = 0
    e = quiet(mf.kernel)
    return e, mf.converged, mf.mo_energy.min(), mf


def one_k(k):
    mol = gto.M(atom=geom, basis=make_basis(k), verbose=0)
    xmol, contr = _decontract_spinor(mol, None)
    n2c = xmol.nao_2c()
    sl = xmol.aoslice_2c_by_atom()
    idx2 = [numpy.arange(p0, p1) for p0, p1 in sl[:, 2:4]]
    idx4 = [numpy.concatenate([i, i + n2c]) for i in idx2]
    m4 = scf.DHF(xmol)
    m4.verbose = 0
    h4, s4 = m4.get_hcore(), m4.get_ovlp()
    s2 = s4[:n2c, :n2c]
    smin = numpy.linalg.eigvalsh(s2).min()
    cc = lambda h: reduce(numpy.dot, (contr.conj().T, h, contr))
    sc = mol.intor('int1e_ovlp_spinor')

    ints = {f: atm_ints(mol, f) for f in (0, 1, 3)}      # DC, Gaunt, Gaunt + gauge
    gm = lambda f, i, four: construct_molecular_matrix(X.extract_ith_integral(ints[f], i), sl, xmol, n2c, four)

    # ---- part A: J/K split of the atomic 4c 2e Fock
    d4 = gm(0, 11, True)
    g4b = gm(0, 5, True)
    jfull = numpy.zeros_like(h4)
    g4lr = jfull.copy()
    errA = 0.
    for i4 in idx4:
        dk = numpy.zeros_like(d4)
        dk[numpy.ix_(i4, i4)] = d4[numpy.ix_(i4, i4)]
        vj, vk = m4.get_jk(xmol, dk, hermi=1)
        errA = max(errA, abs((vj - vk)[numpy.ix_(i4, i4)] - g4b[numpy.ix_(i4, i4)]).max())
        jfull += vj
        g4lr[numpy.ix_(i4, i4)] += g4b[numpy.ix_(i4, i4)] - vj[numpy.ix_(i4, i4)]
    g4lr += jfull

    # ---- part B: amfX2C, three forms of G4c in the decoupling
    x, _, r, _ = X.x2c1e_hfw0_4cmat(h4, s4, mol=xmol)
    h1e = X.to_2c(x, r, h4)
    dfb = X.to_2c(gm(0, 0, False), gm(0, 1, False), g4b) - gm(0, 6, False)
    dfp = projector_form([dfb[numpy.ix_(i, i)] for i in idx2], s2, idx2)
    g4p = projector_form([g4b[numpy.ix_(i, i)] for i in idx4], s4, idx4)
    # SAP: the Coulomb part as a fitted local potential (true molecular operator)
    vs4 = sap_4c(xmol)
    g4sap = vs4.copy()
    for i4 in idx4:
        dk = numpy.zeros_like(d4)
        dk[numpy.ix_(i4, i4)] = d4[numpy.ix_(i4, i4)]
        vj = m4.get_jk(xmol, dk, hermi=1, with_k=False)[0]
        g4sap[numpy.ix_(i4, i4)] += g4b[numpy.ix_(i4, i4)] - vj[numpy.ix_(i4, i4)]
    xs, _, rs, _ = X.x2c1e_hfw0_4cmat(h4 + vs4, s4, mol=xmol)
    hs = {'1e': h1e,
          # SAP-X2C (Surjuse, Valeev): X from V + V_scr, V_scr subtracted untransformed
          'SAPX2C': X.to_2c(xs, rs, h4 + vs4) - vs4[:n2c, :n2c]}
    for key, g4, df in (('block', g4b, dfb), ('proj', g4p, dfp), ('LR', g4lr, dfb), ('LR+dfP', g4lr, dfp), ('SAP+K', g4sap, dfb)):
        xa, _, ra, _ = X.x2c1e_hfw0_4cmat(h4 + g4, s4, mol=xmol)
        hs[key] = X.to_2c(xa, ra, h4) + df
    resB = {}
    for key, h in hs.items():
        hc = cc(h)
        e, conv, emin, mf = scf_run(mol, hc)
        resB[key] = (e, conv, emin, scipy.linalg.eigh(hc, sc, eigvals_only=True).min())
        if key == '1e':
            dm1 = contr @ mf.make_rdm1() @ contr.conj().T

    # ---- part C: pieces of the x2camf SO AMF correction, first order
    so = {f: gm(f, 10, False) for f in (0, 1, 3)}
    pieces = {'SO(DC)': so[0], 'Gaunt': so[1] - so[0], 'gauge': so[3] - so[1]}
    resC = {}
    for key, a in pieces.items():
        ap = projector_form([a[numpy.ix_(i, i)] for i in idx2], s2, idx2)
        resC[key] = (numpy.einsum('ij,ji', a, dm1).real, numpy.einsum('ij,ji', ap, dm1).real,
                     orth_norm(a, s2), orth_norm(ap, s2))
    return smin, errA, resB, resC


print('# %s, %s + k diffuse s,p (ratio 2.5); point nucleus' % (geom, base))
rows = []
for k in range(kmax + 1):
    t0 = time.time()
    try:
        rows.append((k,) + one_k(k))
    except Exception as ex:
        print('# k=%d failed: %r' % (k, ex), flush=True)
        break
    k_, smin, errA, resB, resC = rows[-1]
    print('# k=%d %.0f s  Smin %.2e  |(J-K)[D_K]_KK - G4c_K| = %.1e  aX2C-LR %.6e%s  1st-order Gaunt block/projf '
          '%.6e %.6e' % (k, time.time() - t0, smin, errA, resB['LR'][0] - resB['1e'][0],
                         '' if resB['LR'][1] else '*', resC['Gaunt'][0], resC['Gaunt'][1]), flush=True)

f = lambda x, c=True: ('% .5e' % x) + (' ' if c else '*')
keysB = ['block', 'proj', 'LR', 'LR+dfP', 'SAPX2C', 'SAP+K']
print('\n## amfX2C: E - E(X2C-1e) / Eh   (* = not converged)  [J/K check: max dev of (J-K)[D_K]_KK from atomic G4c_K]')
print('%2s %9s %8s  ' % ('k', 'Smin', 'J/K chk') + ''.join('%-13s' % s for s in keysB))
for k, smin, errA, resB, resC in rows:
    print('%2d %9.2e %8.1e  ' % (k, smin, errA) + ''.join(f(resB[s][0] - resB['1e'][0], resB[s][1]) for s in keysB))
print('\n## amfX2C: lowest h_core eigenvalue / lowest orbital energy / Eh')
print('%2s ' % 'k' + ''.join('%-26s' % s for s in ['1e'] + keysB))
for k, smin, errA, resB, resC in rows:
    print('%2d ' % k + ''.join('%12.5f %12.5f ' % (resB[s][3], resB[s][2]) for s in ['1e'] + keysB))

print('\n## x2camf SO AMF pieces: first-order energy Tr(D_1e dF) block / projector form, |dF|orth block / projf')
print('%2s %9s ' % ('k', 'Smin') + ''.join('%-50s' % s for s in ('SO(DC)', 'Gaunt', 'gauge')))
for k, smin, errA, resB, resC in rows:
    print('%2d %9.2e ' % (k, smin) + ''.join('% .7e % .7e %8.4f %8.4f  ' % resC[s] for s in ('SO(DC)', 'Gaunt', 'gauge')))

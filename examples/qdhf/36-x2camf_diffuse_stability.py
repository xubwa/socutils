#!/usr/bin/env python

'''
Numerical stability of x2camf (X2C-1e decoupling + atom-block-diagonal
spin-orbit AMF correction) with increasingly diffuse basis sets, compared
with X2C-1e, x2cmp and amfX2C (atom-block-diagonal scalar 2e Fock in the
decoupling; block and projector form, cf. 23-amfx2c_projector_form.py).

Basis: <basis> + k even-tempered s, p functions (ratio 2.5) on every atom,
k = 0 .. kmax, until the overlap is nearly singular.

Per k (first table): min eigenvalue of S (contracted / decontracted 2c),
E(DHF) (pyscf 4c Dirac-Coulomb HF, same contracted basis) and E - E(DHF) for

  1e        X2C-1e
  amf       x2camf, block-diagonal SO AMF  (what x2cmp(flavor='x2camf') does)
  p1e-6     x2camf, SO AMF projected onto the overlap eigenvectors > 1e-6
  p1e-5     same, > 1e-5
  amfP      x2camf, SO AMF in projector form  S_.K S_K^-1 A_K S_K^-1 S_K.
  x2cmp     x2cmp(flavor='x2cmp')
  aX2C      amfX2C, block form
  aX2CP     amfX2C, projector form

Second table: the AMF energy contribution E(amf) - E(1e) (and the projected /
projector-form ones), which should not depend on the diffuse functions.

Third table (collapse diagnostics): lowest eigenvalue of the 2c core
Hamiltonian (generalized, contracted basis) and lowest orbital energy of the
converged Fock matrix, per method; and |dF|_orth = max |eig| of the SO AMF
correction in the canonically orthonormalized decontracted basis,
U^H dF U with U = u / sqrt(w), for the block, projected and projector forms.
A spurious state from the block form acting on near-linearly-dependent
cross-atom combinations would show up as |dF|_orth ~ <dF>/w_min and as a
lowest eigenvalue far below the X2C-1e one.

Usage:  python 36-x2camf_diffuse_stability.py kmax geom basis [dhf=1] [gaunt_breit=0]
  basis: one name (unc-<name> = fully uncontracted), or 'H:cc-pvdz,Cl:cc-pvdz'
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
base = sys.argv[3] if len(sys.argv) > 3 else 'cc-pvdz'
do_dhf = int(sys.argv[4]) if len(sys.argv) > 4 else 1
# 1: Gaunt + Breit in the atomic AMF calculation (the .x2camf() default)
gb = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
os.environ['X2CAMF_TIMING'] = '0'
elems = sorted(set(a.split()[0] for a in geom.split(';')))
if ':' in base:
    base = dict(x.split(':') for x in base.split(','))
else:
    base = {el: base for el in elems}

# the atomic x2camf calculation depends only on (Z, basis): cache it, so that
# x2cmp(flavor=...) and the hand-built variants below share one atomic run
_atm = x2camf.libx2camf.atm_integrals
_atm_cache = {}


def atm_integrals_cached(*args, **kwargs):
    key = repr([a.tobytes() if isinstance(a, numpy.ndarray) else a for a in args]) + repr(sorted(kwargs.items()))
    if key not in _atm_cache:
        _atm_cache[key] = _atm(*args, **kwargs)
    return _atm_cache[key]


x2camf.libx2camf.atm_integrals = atm_integrals_cached


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
        # x2camf expects the shells sorted by angular momentum
        bas[el] = sorted(b + extra, key=lambda sh: sh[0])
    return bas


def quiet(f):
    with open(os.devnull, 'w') as fn, contextlib.redirect_stdout(fn):
        return f()


def projector_form(a_blocks, s_full, slices):
    '''sum_K S_{.K} S_K^-1 A_K S_K^-1 S_{K.} (see 23-amfx2c_projector_form.py)'''
    out = numpy.zeros_like(s_full)
    for a, idx in zip(a_blocks, slices):
        sk = s_full[numpy.ix_(idx, idx)]
        m = scipy.linalg.solve(sk, scipy.linalg.solve(sk, a).conj().T).conj().T
        sxk = s_full[:, idx]
        out += sxk @ m @ sxk.conj().T
    return out


def canonical(s, thr=0.):
    w, u = numpy.linalg.eigh(s)
    keep = w > thr
    return u[:, keep] / numpy.sqrt(w[keep])


def orth_norm(a, uc):
    return abs(numpy.linalg.eigvalsh(uc.conj().T @ a @ uc)).max()


def scf_run(mol, hc):
    mf = spinor_hf.SCF(mol)
    mf.get_hcore = lambda *a: hc.copy()
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.verbose = 0
    e = quiet(mf.kernel)
    return e, mf.converged, mf.mo_energy.min()


def one_k(k):
    mol = gto.M(atom=geom, basis=make_basis(k), verbose=0)
    smin_c = numpy.linalg.eigvalsh(mol.intor('int1e_ovlp')).min()

    mf = spinor_hf.SCF(mol)
    mf.x2cmp(flavor='x2cmp', with_gaunt=gb, with_breit=gb, with_pcc=False)
    xo = mf.with_x2c
    h_x2cmp_c = quiet(lambda: xo.get_hcore(mol))       # contracted, also fills the atomic cache
    atm = xo.atomic_integrals
    xmol, contr = _decontract_spinor(mol, xo.xuncontract)
    n2c = xmol.nao_2c()
    sl = xmol.aoslice_2c_by_atom()
    gm = lambda i, four: construct_molecular_matrix(X.extract_ith_integral(atm, i), sl, xmol, n2c, four)
    idx2 = [numpy.arange(p0, p1) for p0, p1 in sl[:, 2:4]]
    idx4 = [numpy.concatenate([i, i + n2c]) for i in idx2]

    m4 = scf.DHF(xmol)
    h4, s4 = m4.get_hcore(), m4.get_ovlp()
    s2 = s4[:n2c, :n2c]
    w2 = numpy.linalg.eigvalsh(s2)
    smin_u = w2.min()
    uc = canonical(s2)

    x, st, r, _ = X.x2c1e_hfw0_4cmat(h4, s4, mol=xmol)
    h1e = X.to_2c(x, r, h4)
    so = gm(10, False)
    so_pf = projector_form([so[numpy.ix_(i, i)] for i in idx2], s2, idx2)
    so_p = {}
    for t in (1e-6, 1e-5):
        u = canonical(s2, t)
        so_p[t] = s2 @ u @ (u.conj().T @ so @ u) @ u.conj().T @ s2

    # amfX2C, block and projector form (as in 23-amfx2c_projector_form.py)
    hax = {}
    g4b = gm(5, True)
    dfb = X.to_2c(gm(0, False), gm(1, False), g4b) - gm(6, False)
    for form in ('block', 'projector'):
        g4, df = g4b, dfb
        if form == 'projector':
            g4 = projector_form([g4b[numpy.ix_(i, i)] for i in idx4], s4, idx4)
            df = projector_form([dfb[numpy.ix_(i, i)] for i in idx2], s2, idx2)
        xa, _, ra, _ = X.x2c1e_hfw0_4cmat(h4 + g4, s4, mol=xmol)
        hax[form] = X.to_2c(xa, ra, h4) + df

    cc = lambda h: reduce(numpy.dot, (contr.conj().T, h, contr))
    hcs = {'1e': cc(h1e), 'amf': cc(h1e + so), 'p1e-6': cc(h1e + so_p[1e-6]),
           'p1e-5': cc(h1e + so_p[1e-5]), 'amfP': cc(h1e + so_pf), 'x2cmp': h_x2cmp_c,
           'aX2C': cc(hax['block']), 'aX2CP': cc(hax['projector'])}
    # consistency: the hand-built block x2camf == x2cmp(flavor='x2camf')
    mfa = spinor_hf.SCF(mol)
    mfa.x2cmp(flavor='x2camf', with_gaunt=gb, with_breit=gb, with_pcc=False)
    dev = abs(quiet(lambda: mfa.with_x2c.get_hcore(mol)) - hcs['amf']).max()

    sc = mol.intor('int1e_ovlp_spinor')
    res = {}
    for key, hc in hcs.items():
        e, conv, emin = scf_run(mol, hc)
        hmin = scipy.linalg.eigh(hc, sc, eigvals_only=True).min()
        res[key] = (e, conv, emin, hmin)

    e4 = numpy.nan
    if do_dhf:
        m4c = scf.DHF(mol)
        m4c.conv_tol = 1e-10
        m4c.verbose = 0
        e4 = quiet(m4c.kernel)
        if not m4c.converged:
            e4 = -abs(e4) * 0 + e4   # keep, flagged below
        res['dhf_conv'] = m4c.converged
        res['dhf_e1s'] = m4c.mo_energy[m4c.mo_occ > 0].min()
    norms = (orth_norm(so, uc), orth_norm(so_p[1e-6], uc), orth_norm(so_pf, uc),
             orth_norm(dfb, uc), orth_norm(hax['block'] - h1e, uc))
    return smin_c, smin_u, e4, res, norms, dev


keys = ['1e', 'amf', 'p1e-6', 'p1e-5', 'amfP', 'x2cmp', 'aX2C', 'aX2CP']
print('# %s, %s + k diffuse s,p (ratio 2.5); point nucleus, no pcc, atomic AMF %s' % (geom, base,
      'Dirac-Coulomb-Breit (Gaunt + gauge)' if gb else 'Dirac-Coulomb'))
rows = []
for k in range(kmax + 1):
    t0 = time.time()
    try:
        rows.append((k,) + one_k(k))
    except Exception as ex:
        print('# k=%d failed: %r' % (k, ex), flush=True)
        break
    k_, smc, smu, e4, res, norms, dev = rows[-1]
    print('# k=%d done in %.0f s; |h(block, here) - h(x2cmp flavor x2camf)|max = %.1e' % (k, time.time() - t0, dev),
          flush=True)
    print('#   Smin %.2e  dE_amf: block %.6e p1e-5 %.6e projf %.6e  |dF|orth: block %.4e projf %.4e  aX2C %.3e%s'
          % (smu, res['amf'][0] - res['1e'][0], res['p1e-5'][0] - res['1e'][0], res['amfP'][0] - res['1e'][0],
             norms[0], norms[2], res['aX2C'][0] - res['1e'][0], '' if res['aX2C'][1] else '*'), flush=True)


def f(x, conv=True):
    return ('% .4e' % x) + (' ' if conv else '*')


print('\n## E - E(DHF) / Eh   (* = SCF not converged)')
print('%2s %9s %9s %17s ' % ('k', 'Smin(c)', 'Smin(2cu)', 'E(DHF)') + ''.join('%-12s' % s for s in keys))
for k, smc, smu, e4, res, norms, dev in rows:
    line = '%2d %9.2e %9.2e %17.9f%s' % (k, smc, smu, e4, '*' if not res.get('dhf_conv', True) else ' ')
    ref = e4 if do_dhf else res['1e'][0]
    for key in keys:
        line += f(res[key][0] - ref, res[key][1])
    print(line)
if not do_dhf:
    print('(no DHF: columns are E - E(X2C-1e))')

print('\n## AMF energy contribution E(x2camf variant) - E(X2C-1e) / Eh')
print('%2s %-12s %-12s %-12s %-12s' % ('k', 'amf', 'p1e-6', 'p1e-5', 'amfP'))
for k, smc, smu, e4, res, norms, dev in rows:
    print('%2d ' % k + ''.join(f(res[key][0] - res['1e'][0]) for key in ('amf', 'p1e-6', 'p1e-5', 'amfP')))

print('\n## lowest eigenvalue of h_core (2c, contracted) / Eh')
print('%2s ' % 'k' + ''.join('%-14s' % s for s in keys))
for k, smc, smu, e4, res, norms, dev in rows:
    print('%2d ' % k + ''.join('%13.6f ' % res[key][3] for key in keys))

print('\n## lowest occupied orbital energy (converged Fock) / Eh   [DHF: e(1s)]')
print('%2s ' % 'k' + ''.join('%-14s' % s for s in keys) + 'DHF')
for k, smc, smu, e4, res, norms, dev in rows:
    print('%2d ' % k + ''.join('%13.6f ' % res[key][2] for key in keys) + ('%13.6f' % res.get('dhf_e1s', numpy.nan)))

print('\n## max |eig| of the correction in the canonically orthonormalized decontracted 2c basis / Eh')
print('%2s %9s %-12s %-12s %-12s %-12s %-12s' % ('k', 'Smin(2cu)', 'SO block', 'SO p1e-6', 'SO projf',
                                                  'aX2C dF', 'aX2C h-h1e'))
for k, smc, smu, e4, res, norms, dev in rows:
    print('%2d %9.2e ' % (k, smu) + ''.join('%11.4e ' % n for n in norms))

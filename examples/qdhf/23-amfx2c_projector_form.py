#!/usr/bin/env python

'''
amfX2C with the atomic two-electron quantities in projector form instead of
atom-block-diagonal form.  An atomic operator represented in the basis of
atom K, A_K (matrix in K's basis), is carried into the molecular basis as

    A_K^mol = S_{.K} S_K^-1 A_K S_K^-1 S_{K.}

(S_{.K}: overlap of all molecular functions with the functions of K).  For
the K-K block this is A_K itself; functions of other atoms that reach into
atom K (diffuse functions) pick up the K-K operator through their overlap,
and near-linearly-dependent cross-atom combinations get ~zero.

Applied to (i) the atomic 4c 2e Fock matrices G4c_K that enter the
decoupling matrix and (ii) the 2c correction dF_K = to_2c_K(G4c_K) - G2c_K.

HF cc-pVDZ + k diffuse s, p;  E - E(DHF).

Usage:  python 23-amfx2c_projector_form.py [kmax] [geom] [basis]
'''

import os
import sys
import contextlib
import importlib.util
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

kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 6
geom = sys.argv[2] if len(sys.argv) > 2 else 'H 0 0 0; F 0 0 0.917'
base = sys.argv[3] if len(sys.argv) > 3 else 'cc-pvdz'
os.environ['X2CAMF_TIMING'] = '0'


def make_basis(k):
    bas = {}
    for el in sorted(set(a.split()[0] for a in geom.split(';'))):
        b = gto.basis.load(base, el)
        raw = gto.mole.uncontracted_basis(b)
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
    '''sum_K S_{.K} S_K^-1 A_K S_K^-1 S_{K.} for atom blocks a_blocks[K]
    (index ranges slices[K] of s_full).'''
    out = numpy.zeros_like(s_full)
    for a, idx in zip(a_blocks, slices):
        sk = s_full[numpy.ix_(idx, idx)]
        m = scipy.linalg.solve(sk, scipy.linalg.solve(sk, a).conj().T).conj().T   # S_K^-1 A S_K^-1
        sxk = s_full[:, idx]
        out += sxk @ m @ sxk.conj().T
    return out


def hcore(mol, form):
    mf = spinor_hf.SCF(mol)
    mf.x2cmp(flavor='x2cmp', with_gaunt=False, with_breit=False, with_pcc=False)
    xo = mf.with_x2c
    atm = {}
    for atom in set(a[0] for a in mol._atom):
        z = elements.charge(gto.mole._std_symbol(atom))
        raw = gto.mole.uncontracted_basis(mol._basis[atom])
        sh = numpy.array([b[0] for b in raw])
        ex = numpy.array([b[-1][0] for b in raw])
        atm[atom] = quiet(lambda: x2camf.libx2camf.atm_integrals(0, z, sh[-1]+1, len(sh), 0, sh, ex,
                                                                  speed_of_light=param.LIGHT_SPEED))
    xmol, _ = xo.get_xmol()
    n2c = xmol.nao_2c()
    sl = xmol.aoslice_2c_by_atom()
    gm = lambda i, four: construct_molecular_matrix(X.extract_ith_integral(atm, i), sl, xmol, n2c, four)
    mf4 = scf.DHF(xmol)
    h4, s4 = mf4.get_hcore(), mf4.get_ovlp()
    g4 = gm(5, True)
    df = X.to_2c(gm(0, False), gm(1, False), g4) - gm(6, False)
    if form == 'projector':
        idx2 = [numpy.arange(p0, p1) for p0, p1 in sl[:, 2:4]]
        idx4 = [numpy.concatenate([i, i + n2c]) for i in idx2]
        g4 = projector_form([g4[numpy.ix_(i, i)] for i in idx4], s4, idx4)
        s2 = s4[:n2c, :n2c]
        df = projector_form([df[numpy.ix_(i, i)] for i in idx2], s2, idx2)
    x, st, r, _ = X.x2c1e_hfw0_4cmat(h4 + g4, s4, mol=xmol)
    h = X.to_2c(x, r, h4) + df
    _, contr = _decontract_spinor(mol, xo.xuncontract)
    hc = reduce(numpy.dot, (contr.conj().T, h, contr))
    xo.get_hcore = lambda *a: hc.copy()
    mf.conv_tol = 1e-11
    mf.verbose = 0
    return quiet(mf.kernel)


print('%s, %s + k diffuse s,p;  E - E(DHF) / Hartree' % (geom, base))
print('%2s %10s %18s  %-16s %-16s' % ('k', 'min eig S', 'E(DHF)', 'amfX2C block', 'amfX2C proj'))
for k in range(kmax + 1):
    mol = gto.M(atom=geom, basis=make_basis(k), verbose=0)
    smin = numpy.linalg.eigvalsh(mol.intor('int1e_ovlp')).min()
    m4 = scf.DHF(mol)
    m4.conv_tol = 1e-10
    m4.verbose = 0
    e4 = quiet(m4.kernel)
    line = '%2d %10.2e %18.10f  ' % (k, smin, e4)
    for form in ('block', 'projector'):
        try:
            line += '% .6e   ' % (hcore(mol, form) - e4)
        except Exception as ex:
            line += '%-16s ' % ('ERR ' + type(ex).__name__)
    print(line, flush=True)

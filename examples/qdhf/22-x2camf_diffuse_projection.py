#!/usr/bin/env python

'''
x2camf with diffuse basis sets: the atom-block-diagonal spin-orbit AMF
correction acts on near-linearly-dependent cross-atom combinations of
diffuse functions.  Projected variant: the correction is restricted to the
span of the eigenvectors of the (decontracted, 2c) overlap with eigenvalue
> thr,
    dF_P = S U (U^H dF U) U^H S.

HBr, cc-pVDZ-DK + k even-tempered s, p functions (ratio 2.5).  Printed:
E(x2camf) - E(X2C-1e), i.e. the energy contribution of the AMF correction,
which should not depend on diffuse functions.

Usage:  python 22-x2camf_diffuse_projection.py [kmax] [basis] [geom]
'''

import os
import sys
import contextlib
import numpy
from functools import reduce
from pyscf import gto, lib
lib.einsum = lambda s, *o, **k: numpy.einsum(s, *o, optimize=True)
from pyscf.x2c.x2c import _decontract_spinor
from socutils.scf import spinor_hf
from socutils.somf import x2cmp as X

kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 6
base = sys.argv[2] if len(sys.argv) > 2 else 'cc-pvdz-dk'
geom = sys.argv[3] if len(sys.argv) > 3 else 'H 0 0 0; Br 0 0 1.414'
thrs = [1e-12, 1e-6, 1e-5, 1e-4]
os.environ['X2CAMF_TIMING'] = '0'


def make_basis(k):
    bas = {}
    for el in sorted(set(a.split()[0] for a in geom.split(';'))):
        b = gto.basis.load(base, el)
        extra = []
        for l in (0, 1):
            emin = min(p[0] for s in b if s[0] == l for p in s[1:])
            extra += [[l, [emin / 2.5**(i+1), 1.]] for i in range(k)]
        # x2camf expects the shells sorted by angular momentum
        bas[el] = sorted(b + extra, key=lambda sh: sh[0])
    return bas


def quiet(f):
    with open(os.devnull, 'w') as fn, contextlib.redirect_stdout(fn):
        return f()


def _so_2c(xo):
    '''atomic spin-orbit AMF matrix that x2cmp (amf_type 'x2camf') added, in
    the decontracted basis, rebuilt from the atomic integrals it stored'''
    from x2camf.x2camf import construct_molecular_matrix
    xmol, _ = xo.get_xmol()
    return construct_molecular_matrix(X.extract_ith_integral(xo.atomic_integrals, 10),
                                      xmol.aoslice_2c_by_atom(), xmol, xmol.nao_2c(), False)


def run(mol, flavor, thr=None):
    mf = spinor_hf.SCF(mol)
    mf.x2cmp(flavor=flavor, with_gaunt=False, with_breit=False, with_pcc=False)
    if thr is not None:
        xo = mf.with_x2c
        hx = quiet(lambda: xo.get_hcore(mol))            # x2camf, contracted
        so_unc = _so_2c(xo)                               # same matrix, decontracted
        xmol, contr = _decontract_spinor(mol, xo.xuncontract)
        s = xmol.intor('int1e_ovlp_spinor')
        w, u = numpy.linalg.eigh(s)
        keep = w > thr
        u = u[:, keep] / numpy.sqrt(w[keep])        # S-orthonormal canonical vectors
        so_p = s @ u @ (u.conj().T @ so_unc @ u) @ u.conj().T @ s
        cc = contr.conj().T
        hc = hx + cc @ (so_p - so_unc) @ contr
        xo.get_hcore = lambda *a: hc.copy()
    mf.conv_tol = 1e-11
    mf.verbose = 0
    e = quiet(mf.kernel)
    return e, mf.converged


print('%s, %s + k diffuse s,p;  E(x2camf) - E(X2C-1e) / Hartree' % (geom, base))
print('%2s %10s %16s %14s ' % ('k', 'min eig S', 'E(X2C-1e)', 'x2camf') +
      ''.join('%-14s' % ('proj %.0e' % t) for t in thrs))
for k in range(kmax + 1):
    mol = gto.M(atom=geom, basis=make_basis(k), verbose=0)
    xmol, _ = _decontract_spinor(mol, None)
    smin = numpy.linalg.eigvalsh(xmol.intor('int1e_ovlp_spinor')).min()
    e1, _ = run(mol, '1e')
    ea, ca = run(mol, 'x2camf')
    line = '%2d %10.2e %16.10f % .6e%s ' % (k, smin, e1, ea - e1, '' if ca else '*')
    for t in thrs:
        ep, cp = run(mol, 'x2camf', t)
        line += '% .6e%s ' % (ep - e1, '' if cp else '*')
    print(line, flush=True)

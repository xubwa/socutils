#!/usr/bin/env python

'''
n-layer rare-gas slab (square layers, lattice constant a, interlayer distance
d, vacuum), SFX2C-1e vs X2C-1e: the macroscopic field of the bare nuclear
sheets and the background,

    F_j = 4 pi sigma (j - (n-1)/2) (1 - n d / Lz),   sigma = Z / a^2,

gives each closed-shell atom the spurious energy -kappa N F_j^2 / c^4 with
kappa = 1/12 (SFX2C-1e, response of X to the field, first order in h) and
kappa = 1/16 (X2C-1e, second-order energy of the spin-orbit interaction with
the field of the bare nuclei, Section 2.4).  The X2C-1e term thus needs the
SCF energy.  On the same density-fitting integrals,

    delta(n) = [E_X2C-1e - E_SFX2C-1e](slab) - n [E_X2C-1e - E_SFX2C-1e](atom)

contains no nonrelativistic finite-size or background Darwin contributions
(both are the same in the two schemes), and is compared with

    pred(n) = sum_j N F_j^2 (1/12 - 1/16) / c^4 = sum_j N F_j^2 / (48 c^4).

The same difference for an atom pair at distance R (field Z/R^2) calibrates
the finite-basis value of this coefficient.  delta(n) is fitted to
n b + k pred(n) (b: per-layer lattice term).

Usage:  python 48-slab_field_so.py [symbol] [c] [nmax]
'''

import sys
import numpy
from pyscf import lib, gto, scf
lib.einsum = numpy.einsum   # pyscf lib.einsum vs numpy 2.x
from pyscf.pbc import gto as pgto, scf as pscf, df as pdf

symb = sys.argv[1] if len(sys.argv) > 1 else 'He'
c = float(sys.argv[2]) if len(sys.argv) > 2 else 10.
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 10
lib.param.LIGHT_SPEED = c
geo = {'He': (5., 5., 30.), 'Ne': (6., 6., 30.)}
a, d, vac = geo[symb]
# even-tempered s (from 0.1) and p; the second-order spin-orbit response to the
# field needs diffuse p functions (unc-cc-pVDZ gives the wrong sign for He)
ET = {'He': (12, 6), 'Ne': (16, 10)}
ns, npp = ET[symb]
basis = {symb: [[0, [x, 1.]] for x in 0.1 * 2.5**numpy.arange(ns)]
         + [[1, [x, 1.]] for x in 0.1 * 2.5**numpy.arange(npp)]}
kmesh = [4, 4, 1]
conv = 1e-11

mol = gto.M(atom='%s 0 0 0' % symb, basis=basis, verbose=0)
Z = N = mol.atom_charge(0)
sigma = Z / a**2
e_atom = (scf.GHF(mol).x2c1e().run(conv_tol=conv).e_tot
          - scf.RHF(mol).sfx2c1e().run(conv_tol=conv).e_tot)
R = 20.
m2 = gto.M(atom=[[symb, (0, 0, 0)], [symb, (0, 0, R)]], basis=basis, unit='B', verbose=0)
e_pair = (scf.GHF(m2).x2c1e().run(conv_tol=conv).e_tot
          - scf.RHF(m2).sfx2c1e().run(conv_tol=conv).e_tot - 2 * e_atom)
k_pair = e_pair / (2 * N * (Z / R**2)**2 / (48 * c**4))
print('%s slab, a = %g, d = %g, vacuum %g bohr, c = %g, even-tempered s%dp%d, k-mesh %s' % (symb, a, d, vac, c, ns, npp, kmesh))
print('isolated atom: E_X2C-1e - E_SFX2C-1e = % .3e' % e_atom)
print('pair R = %g: delta / [2 N F^2/(48 c^4)] = %.4f (this basis)' % (R, k_pair))
print('%3s %7s %14s %14s %14s %8s %10s' % ('n', 'Lz', 'E_SF - E_SF0', 'delta', 'pred', 'ratio', 'F_surf'))
rows = []
for n in range(1, nmax + 1):
    Lz = (n - 1) * d + vac
    atoms = [[symb, (0., 0., j * d)] for j in range(n)]
    cell = pgto.Cell(atom=atoms, a=numpy.diag([a, a, Lz]), basis=basis, unit='B',
                     verbose=0, precision=1e-12).build()
    kpts = cell.make_kpts(kmesh)
    mydf = pdf.GDF(cell, kpts)
    mydf.build()
    e = {}
    for name in ('sf', 'so'):
        if name == 'sf':
            mf = pscf.KRHF(cell, kpts, exxdiv='ewald')
        else:
            mf = pscf.KGHF(cell, kpts, exxdiv='ewald')
        mf.with_df = mydf
        mf = mf.sfx2c1e() if name == 'sf' else mf.x2c1e()
        mf.conv_tol = conv
        e[name] = mf.kernel()
        if not mf.converged:
            print('  warning: %s not converged, n = %d' % (name, n))
    delta = e['so'] - e['sf'] - n * e_atom
    F = [4 * numpy.pi * sigma * (j - (n - 1) / 2.) * (1 - n * d / Lz) for j in range(n)]
    pred = sum(N * f**2 / (48 * c**4) for f in F)
    ratio = (delta - rows[0][1] * n) / pred if n > 1 else float('nan')
    print('%3d %7.1f % .8e % .6e % .6e %8.4f %10.4f' % (n, Lz, e['sf'], delta, pred, ratio, F[-1]), flush=True)
    rows.append((n, delta, pred))
    mydf = mf = None

rows = numpy.array(rows)
A = numpy.c_[rows[:, 0], rows[:, 2]]
b, k = numpy.linalg.lstsq(A, rows[:, 1], rcond=None)[0]
print('fit delta = n b + k pred:  b = % .4e per layer,  k = %.4f  (pair: %.4f)' % (b, k, k_pair))

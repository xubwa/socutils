#!/usr/bin/env python

'''
Background Darwin term of periodic sf-X2C-1e at a real solid density (fcc
rare-gas crystal, c = 137.036 by default).

The Ewald nuclear potential contains the uniform background -rho = -Z/Omega.
X2C-1e picture-changes it with the nuclei; the background inside J is not
transformed, so every electron is shifted by the Darwin term of the nuclear
background,

    dE / atom = -(pi / 2c^2) rho N_e = -(pi / 2c^2) Z^2 / Omega,

independent of the orbital (complete basis, first order).  It is extensive,
does not vanish with k-point or supercell convergence, and depends on volume.

E1 = Tr D_atom (h_X2C - h_NR)(T=0) - [same, isolated atom], with the isolated-
atom NR density and the real-space T = 0 block from a k-mesh average.  In a
cubic (O_h) site the first-order multipole terms vanish for a spherical
density and the bare lattice field is zero, so E1 is the background term.

Usage:  python 32-fcc_jellium_darwin.py [symbol] [a/Angstrom] [kmesh] [c]
'''

import sys
import numpy
from pyscf import lib, gto, scf
lib.einsum = numpy.einsum   # pyscf lib.einsum vs numpy 2.x
from pyscf.pbc import gto as pgto, scf as pscf
from pyscf.pbc.x2c import sfx2c1e as psfx

symb = sys.argv[1] if len(sys.argv) > 1 else 'Ar'
a_ang = float(sys.argv[2]) if len(sys.argv) > 2 else 5.26      # Ar fcc, 5.26 A
nk = int(sys.argv[3]) if len(sys.argv) > 3 else 3
c = float(sys.argv[4]) if len(sys.argv) > 4 else 137.036
lib.param.LIGHT_SPEED = c
basis = {symb: gto.uncontract(gto.load('cc-pvdz', symb))}

mol = gto.M(atom='%s 0 0 0' % symb, basis=basis, verbose=0)
Z = N = mol.atom_charge(0)
m_nr = scf.RHF(mol).run(conv_tol=1e-11)
dm = m_nr.make_rdm1()
e_atom = numpy.einsum('ij,ji', dm, scf.RHF(mol).sfx2c1e().get_hcore() - m_nr.get_hcore())

a = a_ang / 0.529177210903
lat = 0.5 * a * numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
cell = pgto.Cell(atom='%s 0 0 0' % symb, a=lat, basis=basis, unit='B', verbose=0,
                 precision=1e-12).build()
omega = cell.vol
print('%s fcc a = %.3f A, Omega = %.2f bohr^3/atom, c = %g, unc-cc-pVDZ' % (symb, a_ang, omega, c))
print('isolated atom Tr D (h_X2C - h_NR) = %.10f' % e_atom)
for k in sorted({max(1, nk - 1), nk}):
    kpts = cell.make_kpts([k] * 3)
    hx = numpy.asarray(psfx.SpinFreeX2CHelper(cell).get_hcore(kpts=kpts))
    hnr = numpy.asarray(pscf.KRHF(cell, kpts).density_fit().get_hcore(kpts=kpts))
    dh = (hx - hnr).mean(axis=0).real
    e1 = numpy.einsum('ij,ji', dm, dh) - e_atom
    pred = -numpy.pi / (2 * c**2) * Z * N / omega
    print('k-mesh %d^3:  E1 = % .6e Eh/atom   -(pi/2c^2) Z N/Omega = % .6e   ratio %.4f' % (k, e1, pred, e1 / pred), flush=True)

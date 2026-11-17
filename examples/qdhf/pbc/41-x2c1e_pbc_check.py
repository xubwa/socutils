#!/usr/bin/env python
'''
First-order background Darwin term of periodic X2C-1e with spin-orbit coupling
(pyscf.pbc.x2c.x2c1e, spin-orbital basis), the analogue of 34-rs1e_pbc_check.py:

    E1 = Tr D_atom (h - h_NR)(T = 0) - [same for the isolated atom, X2C-1e]

with the isolated-atom NR density on every atom.  Expected: the same
-(pi/2c^2) Z^2 / Omega as sfx2c1e (the spin-dependent part of W does not
contribute to the picture change of the background).

Usage:  python 41-x2c1e_pbc_check.py [symbol] [a/Angstrom] [nk]
'''
import sys
import numpy
import scipy.linalg
from pyscf import lib, gto, scf
lib.einsum = numpy.einsum   # pyscf lib.einsum vs numpy 2.x
from pyscf.pbc import gto as pgto, scf as pscf
from pyscf.pbc.x2c import sfx2c1e as psfx, x2c1e as px2c
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbc_rs1e

symb = sys.argv[1] if len(sys.argv) > 1 else 'Ar'
a_ang = float(sys.argv[2]) if len(sys.argv) > 2 else 5.26
nk = int(sys.argv[3]) if len(sys.argv) > 3 else 2
c = lib.param.LIGHT_SPEED
basis = {symb: gto.uncontract(gto.load('cc-pvdz', symb))}

mol = gto.M(atom='%s 0 0 0' % symb, basis=basis, verbose=0)
Z = mol.atom_charge(0)
m_nr = scf.RHF(mol).run(conv_tol=1e-11)
dm = m_nr.make_rdm1()
dmg = scipy.linalg.block_diag(dm, dm) * .5
hnr_mol = m_nr.get_hcore()
e_atom_sf = numpy.einsum('ij,ji', dm, scf.RHF(mol).sfx2c1e().get_hcore() - hnr_mol)
hg = scf.GHF(mol).x2c1e().get_hcore()
e_atom_so = numpy.einsum('ij,ji', dmg, hg - scipy.linalg.block_diag(hnr_mol, hnr_mol)).real

a = a_ang / 0.529177210903
cell = pgto.Cell(atom='%s 0 0 0' % symb, a=0.5 * a * numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                 basis=basis, unit='B', verbose=0, precision=1e-12).build()
kpts = cell.make_kpts([nk] * 3)
hnr = numpy.asarray(pscf.KRHF(cell, kpts).density_fit().get_hcore(kpts=kpts))
pred = -numpy.pi / (2 * c**2) * Z * Z / cell.vol
print('%s fcc a = %.3f A, Omega = %.2f bohr^3, k-mesh %d^3, unc-cc-pVDZ, c = %.3f' % (symb, a_ang, cell.vol, nk, c))
print('background Darwin -(pi/2c^2) Z^2/Omega = % .6e' % pred)
h = numpy.asarray(psfx.SpinFreeX2CHelper(cell).get_hcore(cell, kpts))
e1 = numpy.einsum('ij,ji', dm, (h - hnr).mean(axis=0).real) - e_atom_sf
print('%-10s E1 = % .6e Eh/atom   E1/pred = % .4f' % ('sfx2c1e', e1, e1 / pred), flush=True)
h = numpy.asarray(px2c.SpinOrbitalX2C1EHelper(cell).get_hcore(cell, kpts))
hnrg = numpy.array([scipy.linalg.block_diag(x, x) for x in hnr])
e1 = numpy.einsum('ij,ji', dmg, (h - hnrg).mean(axis=0)).real - e_atom_so
print('%-10s E1 = % .6e Eh/atom   E1/pred = % .4f' % ('x2c1e', e1, e1 / pred), flush=True)
h = numpy.asarray(pbc_rs1e.SpinOrbitalRS1eHelper(cell).get_hcore(cell, kpts))
print('rs1e (SO) hermiticity %.1e' % abs(h - h.conj().transpose(0, 2, 1)).max())
e1 = numpy.einsum('ij,ji', dmg, (h - hnrg).mean(axis=0)).real - e_atom_so
print('%-10s E1 = % .6e Eh/atom   E1/pred = % .4f' % ('rs1e (SO)', e1, e1 / pred), flush=True)
# isolated atom: rs1e must reduce to X2C-1e
ma = pgto.Cell(atom='%s 0 0 0' % symb, a=numpy.eye(3) * 60, basis=basis, unit='B', verbose=0).build()
hl = pbc_rs1e.SpinOrbitalRS1eHelper(ma)
xm = hl.get_xmol(ma)[0]
z3 = numpy.zeros(3)
vf, wf, sf = (x[0].real for x in pbc_rs1e.sr_nuclear_kpts(mol, z3, 0., with_so=True))
print('molecular X2C-1e from sr_nuclear_kpts(omega=0) vs pyscf: %.1e' % abs(
    pbc_rs1e._x2c_so(mol.intor('int1e_kin'), vf, wf, sf, mol.intor('int1e_ovlp'), c) - hg).max())
print('atom: Tr D (h_X2C - h_NR)  sf % .8f  so % .8f' % (e_atom_sf, e_atom_so))

#!/usr/bin/env python
'''
Check of the periodic spin-free rs1e (pbc_rs1e.py) against pyscf.pbc sfx2c1e,
first order with the isolated-atom NR density on every atom and the real-space
T = 0 block of h from a k-mesh average (as in 32-fcc_jellium_darwin.py):

    E1 = Tr D_atom (h - h_NR)(T = 0) - [same for the isolated atom, sf-X2C-1e]

sfx2c1e should give the background Darwin term -(pi/2c^2) Z^2 / Omega,
rs1e should give ~0 (only short-range overlap / multipole terms).

Usage:  python 34-rs1e_pbc_check.py [symbol] [a/Angstrom] [nk]
'''
import sys, os
import numpy
from pyscf import lib, gto, scf
lib.einsum = numpy.einsum   # pyscf lib.einsum vs numpy 2.x
from pyscf.pbc import gto as pgto, scf as pscf
from pyscf.pbc.x2c import sfx2c1e as psfx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbc_rs1e

symb = sys.argv[1] if len(sys.argv) > 1 else 'Ar'
a_ang = float(sys.argv[2]) if len(sys.argv) > 2 else 5.26
nk = int(sys.argv[3]) if len(sys.argv) > 3 else 3
c = lib.param.LIGHT_SPEED
basis = {symb: gto.uncontract(gto.load('cc-pvdz', symb))}

mol = gto.M(atom='%s 0 0 0' % symb, basis=basis, verbose=0)
Z = mol.atom_charge(0)
m_nr = scf.RHF(mol).run(conv_tol=1e-11)
dm = m_nr.make_rdm1()
e_atom = numpy.einsum('ij,ji', dm, scf.RHF(mol).sfx2c1e().get_hcore() - m_nr.get_hcore())

a = a_ang / 0.529177210903
cell = pgto.Cell(atom='%s 0 0 0' % symb, a=0.5 * a * numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                 basis=basis, unit='B', verbose=0, precision=1e-12).build()
kpts = cell.make_kpts([nk] * 3)
hnr = numpy.asarray(pscf.KRHF(cell, kpts).density_fit().get_hcore(kpts=kpts))
pred = -numpy.pi / (2 * c**2) * Z * Z / cell.vol
print('%s fcc a = %.3f A, Omega = %.2f bohr^3, k-mesh %d^3, unc-cc-pVDZ, c = %.3f' % (symb, a_ang, cell.vol, nk, c))
print('background Darwin -(pi/2c^2) Z^2/Omega = % .6e' % pred)
for name, hlp in (('sfx2c1e', psfx.SpinFreeX2CHelper(cell)),
                  ('rs1e w=1', pbc_rs1e.SpinFreeRS1eHelper(cell)),
                  ('rs1e w=1 no self', None)):
    if hlp is None:
        hlp = pbc_rs1e.SpinFreeRS1eHelper(cell)
        hlp.self_lr = False
    h = numpy.asarray(hlp.get_hcore(cell, kpts))
    dh = (h - hnr).mean(axis=0).real
    e1 = numpy.einsum('ij,ji', dm, dh) - e_atom
    print('%-18s E1 = % .6e Eh/atom   E1/pred = % .4f' % (name, e1, e1 / pred), flush=True)

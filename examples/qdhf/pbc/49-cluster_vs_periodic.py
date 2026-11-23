#!/usr/bin/env python
'''
Background Darwin term as the difference between periodic and finite
boundary conditions within X2C-1e (no four-component calculation needed).

E1 = Tr D_atom (h_X2C - h_NR)_{AA} - [same for the isolated atom], with the
isolated-atom NR density, for (i) the central atom of fcc clusters of
increasing size (molecular SFX2C-1e of the whole cluster) and (ii) an atom of
the periodic crystal (41-x2c1e_pbc_check.py: -1.1041e-4 Eh for Ar, a = 5.26 A,
unc-cc-pVDZ, 2^3 k).  The exact theory does not depend on the boundary
conditions in the bulk limit, so E1(periodic) - E1(cluster centre) is the
part of the X2C-1e error caused by the periodic treatment.  Expected:
-(pi/2c^2) Z^2 / Omega = -1.1038e-4 Eh.

Usage:  python 49-cluster_vs_periodic.py [symbol] [a/Angstrom] [shell radii ...]
'''
import sys
import time
import numpy
from pyscf import lib, gto, scf
lib.einsum = numpy.einsum

symb = sys.argv[1] if len(sys.argv) > 1 else 'Ar'
a_ang = float(sys.argv[2]) if len(sys.argv) > 2 else 5.26
radii = [float(x) for x in sys.argv[3:]] or [1.0, 1.5, 2.0]   # in units of a
c = lib.param.LIGHT_SPEED
basis = {symb: gto.uncontract(gto.load('cc-pvdz', symb))}
a = a_ang / 0.529177210903
mol = gto.M(atom='%s 0 0 0' % symb, basis=basis, verbose=0)
Z = mol.atom_charge(0)
m_nr = scf.RHF(mol).run(conv_tol=1e-11)
dm = m_nr.make_rdm1()
e_atom = numpy.einsum('ij,ji', dm, scf.RHF(mol).sfx2c1e().get_hcore() - m_nr.get_hcore())
na = mol.nao
omega = a**3 / 4
print('%s fcc a = %.3f A, unc-cc-pVDZ, c = %.3f;  -(pi/2c^2) Z^2/Omega = % .5e' % (
    symb, a_ang, c, -numpy.pi / (2 * c**2) * Z**2 / omega))
print('%8s %6s %7s %14s %8s' % ('R/a', 'atoms', 'nao', 'E1 centre', 'time'))
n = int(max(radii)) + 2
pts = [numpy.array([i, j, k]) * .5 * a for i in range(-2 * n, 2 * n + 1)
       for j in range(-2 * n, 2 * n + 1) for k in range(-2 * n, 2 * n + 1) if (i + j + k) % 2 == 0]
for r in radii:
    t0 = time.time()
    sel = sorted([p for p in pts if numpy.linalg.norm(p) <= r * a + 1e-6], key=numpy.linalg.norm)
    cl = gto.M(atom=[[symb, p] for p in sel], basis=basis, unit='B', verbose=0)
    dh = scf.RHF(cl).sfx2c1e().get_hcore()[:na, :na] - cl.intor('int1e_kin')[:na, :na] - cl.intor('int1e_nuc')[:na, :na]
    e1 = numpy.einsum('ij,ji', dm, dh) - e_atom
    print('%8.2f %6d %7d % .5e %8.0f' % (r, len(sel), cl.nao, e1, time.time() - t0), flush=True)

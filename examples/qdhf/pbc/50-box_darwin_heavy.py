#!/usr/bin/env python
'''
Rare-gas atom in a cubic box of edge L (Gamma point): first-order background
Darwin term of periodic SFX2C-1e and X2C-1e from He to Rn,

    E1 = Tr D (h - h_NR)(box) - Tr D (h - h_NR)(isolated atom),

with the relativistic density D of the free atom (sf-X2C-1e RHF / X2C-1e GHF),
compared with -(pi/2c^2) Z N / L^3 (Eq. bgdarwin, first order in 1/c^2).
The isolated atom carries no background, so E1 is the deviation caused by the
periodic treatment.  dyall-v2z (uncontracted), point nuclei.

Usage:  python 50-box_darwin_heavy.py [L] [symbols...]
'''
import sys
import numpy
import scipy.linalg
from pyscf import lib, gto, scf
lib.einsum = numpy.einsum   # pyscf lib.einsum vs numpy 2.x
from pyscf.pbc import gto as pgto, df as pdf
from pyscf.pbc.x2c import sfx2c1e as psfx, x2c1e as px2c

L = float(sys.argv[1]) if len(sys.argv) > 1 else 12.
syms = sys.argv[2:] or ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']
c = lib.param.LIGHT_SPEED


def g2(x):
    return scipy.linalg.block_diag(x, x)


print('atom in a cubic box, L = %g bohr, Gamma point, dyall-v2z, c = %.3f' % (L, c))
print('ratio = E1 / (-(pi/2c^2) Z N / L^3)')
print('%-4s %4s %14s %10s %10s' % ('atom', 'Z', 'formula (Eh)', 'SFX2C-1e', 'X2C-1e'))
for symb in syms:
    basis = {symb: gto.uncontract(gto.load('dyall-v2z', symb))}
    mol = gto.M(atom='%s 0 0 0' % symb, basis=basis, verbose=0)
    Z = N = mol.atom_charge(0)
    hnr_mol = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    m_sf = scf.RHF(mol).sfx2c1e().run(conv_tol=1e-10)
    m_so = scf.GHF(mol).x2c1e().run(conv_tol=1e-10)
    dm, dmg = m_sf.make_rdm1(), m_so.make_rdm1()
    ref_sf = numpy.einsum('ij,ji', dm, m_sf.get_hcore() - hnr_mol)
    ref_so = numpy.einsum('ij,ji', dmg, m_so.get_hcore() - g2(hnr_mol)).real
    cell = pgto.Cell(atom='%s 0 0 0' % symb, a=numpy.eye(3) * L, basis=basis, unit='B',
                     verbose=0, precision=1e-12).build()
    k0 = numpy.zeros((1, 3))
    hnr = cell.pbc_intor('int1e_kin', kpts=k0)[0] + pdf.GDF(cell, k0).get_nuc(k0)[0]
    h_sf = psfx.SpinFreeX2CHelper(cell).get_hcore(cell, k0)[0]
    h_so = px2c.SpinOrbitalX2C1EHelper(cell).get_hcore(cell, k0)[0]
    e_sf = numpy.einsum('ij,ji', dm, h_sf - hnr).real - ref_sf
    e_so = numpy.einsum('ij,ji', dmg, h_so - g2(hnr)).real - ref_so
    pred = -numpy.pi / (2 * c**2) * Z * N / L**3
    print('%-4s %4d % 14.6e %10.4f %10.4f' % (symb, Z, pred, e_sf / pred, e_so / pred), flush=True)

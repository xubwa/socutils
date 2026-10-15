#!/usr/bin/env python

'''
Periodic sf-X2C-1e for an n-layer slab (square He layers, lattice constant a,
interlayer distance d, vacuum, 3D cell of height Lz).

The bare nuclear sheets of a slab produce a macroscopic field that grows with
the number of layers.  With the uniform background of the Ewald potential the
field at layer j (j = 0 .. n-1) is

    F_j = 4 pi sigma (j - (n-1)/2) (1 - n d / Lz),   sigma = Z / a^2.

X2C-1e builds X in this bare field, while the electrons that screen it (J)
are untransformed, so each atom picks up the universal second-order picture-
change term of a uniform field, -N_e F^2 / (12 c^4) (the R^-4 term of
26-universal_r4.py with Z_B/R^2 -> F).  On top of that comes the background
Darwin term of 30-box_jellium_darwin.py, -(pi/2c^2) rho N_e per cell with
rho = n Z / (a^2 Lz), calibrated here against the box at the same c and basis.

    E1 = Tr D_atoms (h_X2C - h_NR)(T=0) per cell, minus n times the isolated
    atom, with the isolated-atom NR density on every atom (no SCF finite-size
    effects) and the real-space T = 0 block of h from an in-plane k-mesh,
    E1 - E_jellium  vs  sum_j -N_e F_j^2 / (12 c^4)

Usage:  python 31-slab_field.py [c]
'''

import sys
import numpy
from pyscf import lib, gto, scf
lib.einsum = numpy.einsum   # pyscf lib.einsum vs numpy 2.x
from pyscf.pbc import gto as pgto, scf as pscf
from pyscf.pbc.x2c import sfx2c1e as psfx

c = float(sys.argv[1]) if len(sys.argv) > 1 else 10.
lib.param.LIGHT_SPEED = c
symb, Z, N = 'He', 2, 2
basis = {symb: gto.uncontract(gto.load('cc-pvdz', symb))}
a, d, vac = 5., 5., 30.
kmesh = [4, 4, 1]
sigma = Z / a**2


def h_diff_t0(cell, kmesh):
    """Real-space T = 0 block of h_X2C - h_NR (k-average over kmesh)."""
    kpts = cell.make_kpts(kmesh)
    hx = psfx.SpinFreeX2CHelper(cell).get_hcore(kpts=kpts)
    hnr = pscf.KRHF(cell, kpts).density_fit().get_hcore(kpts=kpts)
    return (numpy.asarray(hx) - numpy.asarray(hnr)).mean(axis=0).real


mol = gto.M(atom='%s 0 0 0' % symb, basis=basis, verbose=0)
m_nr = scf.RHF(mol).run(conv_tol=1e-11)
dm_atom = m_nr.make_rdm1()
e_atom = numpy.einsum('ij,ji', dm_atom, scf.RHF(mol).sfx2c1e().get_hcore() - m_nr.get_hcore())
na = mol.nao


def e1(cell, kmesh):
    """Tr D_atoms (h_X2C - h_NR) with the isolated-atom NR density on every atom."""
    dh = h_diff_t0(cell, kmesh)
    return sum(numpy.einsum('ij,ji', dm_atom, dh[i*na:(i+1)*na, i*na:(i+1)*na])
               for i in range(cell.natm))


# background Darwin calibration: one atom in a cube, per electron per unit rho
Lb = 16.
box = pgto.Cell(atom='%s 0 0 0' % symb, a=numpy.eye(3) * Lb, basis=basis, unit='B',
                verbose=0, precision=1e-12).build()
f_jell = (e1(box, [1, 1, 1]) - e_atom) / (N * Z / Lb**3)
# finite-basis coefficient of the universal R^-4 term: He2 at R = 20 bohr
R = 20.
m2 = gto.M(atom=[[symb, (0, 0, 0)], [symb, (0, 0, R)]], basis=basis, unit='B', verbose=0)
dh2 = scf.RHF(m2).sfx2c1e().get_hcore() - scf.RHF(m2).get_hcore()
e2 = sum(numpy.einsum('ij,ji', dm_atom, dh2[i*na:(i+1)*na, i*na:(i+1)*na]) for i in range(2)) - 2 * e_atom
kappa_pair = e2 / (-2 * N * Z**2 / (12 * c**4 * R**4))
print('%s slab, a = %g, d = %g, vacuum %g bohr, c = %g, unc-cc-pVDZ, k-mesh %s' % (symb, a, d, vac, c, kmesh))
print('background Darwin per electron per unit rho: %.6e  (CBS -pi/2c^2 = %.6e)' % (f_jell, -numpy.pi / 2 / c**2))
print('He2 pair: R^-4 coefficient / universal value = %.4f (this basis)' % kappa_pair)
print('%3s %7s %12s %12s %12s %12s %8s %10s' % ('n', 'Lz', 'E1', 'E_jell', 'E1-E_jell', 'pred F^2', 'ratio', 'F_surf'))
rows = []
for n in range(1, 11):
    Lz = (n - 1) * d + vac
    atoms = [[symb, (0., 0., j * d)] for j in range(n)]
    cell = pgto.Cell(atom=atoms, a=numpy.diag([a, a, Lz]), basis=basis, unit='B',
                     verbose=0, precision=1e-12).build()
    val = e1(cell, kmesh) - n * e_atom
    rho = n * Z / (a * a * Lz)
    ejell = f_jell * N * n * rho
    F = [4 * numpy.pi * sigma * (j - (n - 1) / 2.) * (1 - n * d / Lz) for j in range(n)]
    pred = sum(-N * f**2 / (12 * c**4) for f in F)
    ratio = (val - ejell) / pred if n > 1 else float('nan')
    print('%3d %7.1f % .5e % .5e % .5e % .5e %8.4f %10.4f' % (n, Lz, val, ejell, val - ejell, pred, ratio, F[-1]), flush=True)
    rows.append((n, val - ejell, pred))

# residual = n * (per-layer lattice term) + kappa * sum_j -N F_j^2 / (12 c^4)
rows = numpy.array(rows)
A = numpy.c_[rows[:, 0], rows[:, 2]]
b, kappa = numpy.linalg.lstsq(A, rows[:, 1], rcond=None)[0]
print('fit E1 - E_jell = n b + kappa * pred:  b = %.4e per layer,  kappa = %.4f  (pair: %.4f)' % (b, kappa, kappa_pair))

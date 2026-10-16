#!/usr/bin/env python
'''
Equation of state of a rock-salt crystal with periodic HF: pyscf.pbc sfx2c1e
(X from the full Ewald nuclear potential) against rs1e (pbc_rs1e.py), with one
GDF per volume shared by both, so the density-fitting error cancels in the
difference.  Expected: E_sfx2c1e - E_rs1e = -(pi/2c^2) Z_cell^2 / Omega + const,
i.e. a spurious pressure -A/Omega^2 and a smaller lattice constant.

Usage:  python 35-rocksalt_eos.py [A] [B] [a_min a_max npts] [nk]
'''
import sys, os, time, gc
import numpy
from pyscf import lib, gto
lib.einsum = numpy.einsum   # pyscf lib.einsum vs numpy 2.x
from pyscf.pbc import gto as pgto, scf as pscf, df as pdf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbc_rs1e

A = sys.argv[1] if len(sys.argv) > 1 else 'Na'
B = sys.argv[2] if len(sys.argv) > 2 else 'Cl'
a0, a1, npt = (float(x) for x in sys.argv[3:6]) if len(sys.argv) > 5 else (5.4, 6.0, 7)
nk = int(sys.argv[6]) if len(sys.argv) > 6 else 2
c = lib.param.LIGHT_SPEED
t0 = time.time()


def trimmed(symb, emin=0.1):
    '''cc-pVDZ (dyall-v2z where pyscf has no cc-pVDZ) without primitives < emin
    (diffuse functions are linearly dependent in a dense ionic solid); shells
    left empty are dropped.'''
    out = []
    try:
        shells = gto.load('cc-pvdz', symb)
    except Exception:
        shells = gto.load('dyall-v2z', symb)
    for sh in shells:
        head = sh[:2] if isinstance(sh[1], int) else sh[:1]     # [l] or [l, kappa]
        prims = [p for p in sh[len(head):] if p[0] >= emin]
        if prims and any(abs(c) > 1e-8 for p in prims for c in p[1:]):
            out.append(list(head) + prims)
    return out


basis = {A: trimmed(A), B: trimmed(B)}
print('%s%s rock salt, HF, cc-pVDZ/dyall-v2z (exponents < 0.1 dropped), k-mesh %d^3, c = %.3f' % (A, B, nk, c))
print('%8s %10s %18s %18s %14s %14s %8s' % ('a/A', 'Omega', 'E sfx2c1e', 'E rs1e', 'dE', '-(pi/2c^2)Zc^2/O', 'time'))
zc = None
rows = []
for a_ang in numpy.linspace(a0, a1, int(npt)):
    a = a_ang / 0.529177210903
    lat = 0.5 * a * numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    cell = pgto.Cell(atom=[[A, (0, 0, 0)], [B, (a / 2, 0, 0)]], a=lat, basis=basis, unit='B',
                     verbose=0, precision=1e-10).build()
    zc = cell.atom_charges().sum()
    kpts = cell.make_kpts([nk] * 3)
    mydf = pdf.GDF(cell, kpts)
    mydf.build()
    e = {}
    dm0 = None
    for name in ('sfx2c1e', 'rs1e'):
        mf = pscf.KRHF(cell, kpts, exxdiv='ewald')
        mf.with_df = mydf
        mf = mf.sfx2c1e() if name == 'sfx2c1e' else pbc_rs1e.sfrs1e(mf)
        mf.conv_tol = 1e-10
        e[name] = mf.kernel(dm0)
        dm0 = mf.make_rdm1()
        if not mf.converged:
            print('  warning: %s not converged at a = %.3f' % (name, a_ang))
    # free the GDF integrals (one temporary HDF5 file per volume, ~10 GB for
    # uncontracted all-electron heavy-atom bases) before the next volume
    del mf, dm0
    mydf = None
    gc.collect()
    pred = -numpy.pi / (2 * c**2) * zc**2 / cell.vol
    rows.append((a_ang, cell.vol, e['sfx2c1e'], e['rs1e']))
    print('%8.4f %10.3f %18.10f %18.10f % .6e % .6e %8.0f' % (
        a_ang, cell.vol, e['sfx2c1e'], e['rs1e'], e['sfx2c1e'] - e['rs1e'], pred, time.time() - t0), flush=True)

rows = numpy.array(rows)
om = rows[:, 1]
d = rows[:, 2] - rows[:, 3]
# fit dE = const - A_fit / Omega
M = numpy.c_[numpy.ones_like(om), -1 / om]
const, afit = numpy.linalg.lstsq(M, d, rcond=None)[0]
print('fit dE = const - A/Omega:  A = %.6e  (background Darwin: (pi/2c^2) Z^2 = %.6e, ratio %.4f)  const = %.3e' % (
    afit, numpy.pi / (2 * c**2) * zc**2, afit / (numpy.pi / (2 * c**2) * zc**2), const))


def eos_min(e):
    p = numpy.polyfit(rows[:, 0], e, 4)
    r = [x.real for x in numpy.roots(numpy.polyder(p)) if abs(x.imag) < 1e-9 and rows[0, 0] <= x.real <= rows[-1, 0]]
    if not r:
        return None, None
    am = r[0]
    # B = V d2E/dV2 at the minimum; V = a^3/4 (A^3 per formula unit), dE/da = 0
    # there, so d2E/dV2 = E''(a) / (dV/da)^2; 1 Eh/A^3 = 4359.74 GPa
    vb = am**3 / 4
    bmod = vb * numpy.polyval(numpy.polyder(p, 2), am) / (3 * am**2 / 4)**2 * 4359.744
    return am, bmod


for name, col in (('sfx2c1e', 2), ('rs1e', 3)):
    am, bm = eos_min(rows[:, col])
    if am is not None:
        print('%-8s a_eq = %.5f A   B = %.2f GPa' % (name, am, bm))

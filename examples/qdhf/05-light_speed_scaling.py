#!/usr/bin/env python

'''
Scaling of the spurious sf-X2C-1e long-range tails with the speed of light.

The tails come from the picture-change difference of the charge
distribution, which is O(1/c^2).  Hence c^2 R^n dE(R) should be independent
of c:

    HF...He   n = 2   (charge - picture-change dipole)
    He2...He  n = 3   (charge - picture-change quadrupole)
    He...He   n = 4   (field-squared polarization of the decoupling)

Result (cc-pVDZ, 05-light_speed_scaling.cc-pvdz.out): the dipole (R^-2) and
quadrupole (R^-3) terms scale as 1/c^2, with the same c -> infinity limit for
'1e' and 'atom1e'.  The field-squared R^-4 term of He...He scales as 1/c^4
(c^2 R^4 dE drops by 4 when c doubles; c^4 R^4 dE = -1.624 ('1e'),
-0.080 ('atom1e')).

The speed of light is passed as a string and parsed in quad precision; the
monomers and the complex use the same c.
'''

from pyscf import gto, lib
from socutils import qdhf
from socutils.qdhf.qarray import qmp

basis = 'cc-pvdz'
bohr = 1 / 0.52917721092
c0 = qmp.mpf(lib.param.LIGHT_SPEED)
scales = [1, 2, 4, 8]
distances = [160, 320]

systems = [
    ('HF...He', 'F 0 0 0; H 0 0 0.917', 'He 0 0 {r}', 2),
    ('He2...He', 'He 0 0 -1.5; He 0 0 1.5', 'He 0 0 {r}', 3),
    ('He...He', 'He 0 0 0', 'He 0 0 {r}', 4),
]

def energy(geom, c, approx):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    mf = qdhf.SFX2C1E_RHF(mol, light_speed=c)
    mf.approx = approx
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-17
    mf.max_cycle = 100
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        print('WARNING: not converged', geom, c)
    return e

for approx in ('1e', 'atom1e'):
    for label, geom_a, geom_b, n in systems:
        print('\nsf-X2C-1e(%s)  %s   c^2 R^%d dE  (a.u.)' % (approx, label, n))
        print('   c/c0    R=160 A                 R=320 A')
        for k in scales:
            c = qmp.nstr(c0 * k, 34)
            ea = energy(geom_a, c, approx)
            eb = energy(geom_b.format(r=0), c, approx)
            row = []
            for r in distances:
                e = energy(geom_a + '; ' + geom_b.format(r=r), c, approx)
                de = e - ea - eb
                row.append(de * (qmp.mpf(c))**2 * (qmp.mpf(r) * bohr)**n)
            print('%6d   % .10e   % .10e' % (k, float(row[0]), float(row[1])), flush=True)

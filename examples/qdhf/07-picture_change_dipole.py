#!/usr/bin/env python

'''
The R^-2 coefficient of the spurious sf-X2C-1e tail of a dipolar molecule A
interacting with a far neutral atom B (charge Z_B on the +z axis) is a
property of A alone.  With approx='atom1e', X and R of A do not depend on
V_B, and V_B enters only through its picture-change transformed form, so by
the Hellmann-Feynman theorem

    lim R^2 dE(R) = Z_B * dmu_z,
    dmu_z = -Tr[D_A (PC(z) - z)],   PC(z) = R^T (z + X^T <p chi|z|p chi> X / (4c^2)) R

i.e. Z_B times the picture-change error of the electronic dipole moment of
A in the finite basis (in a complete basis PC(z) - z is O(1/c^4)).  This
needs only the monomer and allows large basis sets, e.g. even-tempered
(ETB) sets approaching the basis-set limit.

Everything is in quadruple precision: the monomer SCF (integral-direct
J/K, so large basis sets fit in memory), the X2C X and R matrices, and the
dipole integrals <chi|z|chi>, <p chi|z|p chi>.  The near linear
dependence of dense even-tempered sets is harmless at this precision.

Usage:  python 07-picture_change_dipole.py
'''

import sys
import time
import numpy
from pyscf import gto
from socutils.qdhf import x2c, qarray
from socutils.qdhf.qarray import qmp

def delta_mu(mol, approx='atom1e'):
    mf = x2c.SFX2C1E_RHF(mol)
    mf.approx = approx
    mf.direct = True
    mf.conv_tol = 1e-26
    mf.conv_tol_grad = 1e-13
    mf.lindep = 1e-28
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        print('WARNING: SCF not converged')
    dm = mf.make_rdm1()
    xmol, contr = x2c.uncontract_mol(mol)
    d = x2c.get_xr(xmol, approx=approx)
    qi, x, r, c = d['qi'], d['x'], d['r'], d['c']
    z = qi.dip_x(2)
    dy = qi.dip_y(2)
    prp = dy[0::3, 0::3] + dy[1::3, 1::3] + dy[2::3, 2::3]
    pc = r.T @ (z + x.T @ prp @ x * (1 / (4 * c**2))) @ r
    dmx = contr @ dm @ contr.T
    return -qmp.re(qarray.trace_dot(dmx, pc - z)), mol.nao, e

def etb_basis(beta, lmax_f=3, lmax_h=2):
    '''even-tempered basis spanning a fixed exponent range for each l'''
    ranges_f = [(0.05, 5e4), (0.04, 200.), (0.1, 10.), (0.2, 5.), (0.3, 3.)]
    ranges_h = [(0.02, 100.), (0.05, 5.), (0.1, 3.), (0.2, 2.)]
    def make(ranges, lmax):
        etbs = []
        for l in range(lmax+1):
            a0, a1 = ranges[l]
            nprim = int(numpy.floor(numpy.log(a1/a0) / numpy.log(beta))) + 1
            etbs.append((l, nprim, a0, beta))
        return gto.etbs(etbs)
    return {'F': make(ranges_f, lmax_f), 'H': make(ranges_h, lmax_h)}

if __name__ == '__main__':
    geom = 'F 0 0 0; H 0 0 0.917'
    print('HF molecule, sf-X2C-1e(atom1e): picture-change error of the electronic dipole (quad)')
    print('%-30s %5s   %-26s  %-12s %s' % ('basis', 'nao', 'dmu_z (a.u.)', 'Z_He*dmu_z', 'time/s'))
    bases = [('cc-pVDZ', 'cc-pvdz'), ('cc-pVTZ', 'cc-pvtz'), ('aug-cc-pVTZ', 'aug-cc-pvtz'),
             ('cc-pVQZ', 'cc-pvqz')]
    for beta in (3.0, 2.5, 2.0, 1.8):
        bases.append(('ETB beta=%.1f (F spdf, H spd)' % beta, etb_basis(beta)))
    for beta in (2.5, 2.0):
        bases.append(('ETB beta=%.1f (F spdfg, H spdf)' % beta, etb_basis(beta, 4, 3)))
    # sp only: T maps an s (p) Gaussian on a center to s (p) functions on the
    # same center, so completeness of the one-center sp radial spaces is
    # enough for <T mu| S^-1 |V nu> = <T mu|V nu>, i.e. dmu -> 0
    for beta in (3.0, 2.5, 2.0, 1.7, 1.5):
        bases.append(('ETB beta=%.1f (sp)' % beta, etb_basis(beta, 1, 1)))
    if len(sys.argv) > 1:
        bases = [b for b in bases if any(k in b[0] for k in sys.argv[1:])]
    for label, basis in bases:
        t0 = time.time()
        mol = gto.M(atom=geom, basis=basis, verbose=0)
        dmu, nao, e = delta_mu(mol)
        print('%-30s %5d   %-26s  % .6e  %.0f' % (label, nao, qmp.nstr(dmu, 16), float(2 * dmu),
                                                 time.time() - t0), flush=True)

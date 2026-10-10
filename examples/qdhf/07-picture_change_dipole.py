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

The density matrix is taken from the double precision pyscf sf-X2C-1e(atom1e)
RHF (sufficient for a first-order property); X and R come from socutils.qdhf.

Usage:  python 07-picture_change_dipole.py
'''

import numpy
from pyscf import gto, scf, lib
from socutils.qdhf import x2c

def delta_mu(mol, approx='atom1e'):
    mf = scf.RHF(mol).sfx2c1e()
    mf.with_x2c.approx = approx
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    dm = mf.make_rdm1()
    xmol, contr = x2c.uncontract_mol(mol)
    d = x2c.get_xr(xmol, approx=approx)
    x = d['x'].to_numpy()
    r = d['r'].to_numpy()
    c = float(d['c'])
    contr = contr.to_numpy()
    n = xmol.nao
    with xmol.with_common_orig((0, 0, 0)):
        z = xmol.intor_symmetric('int1e_r')[2]
        prp = xmol.intor_symmetric('int1e_sprsp').reshape(3, 4, n, n)[2, 3]
    pc = r.T @ (z + x.T @ prp @ x / (4 * c**2)) @ r
    dmx = contr @ dm @ contr.T
    return -numpy.einsum('ij,ji->', dmx, pc - z), mol.nao, mf.e_tot

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
    print('HF molecule, sf-X2C-1e(atom1e): picture-change error of the electronic dipole')
    print('%-28s %5s   %-14s  %-14s' % ('basis', 'nao', 'dmu_z (a.u.)', 'Z_He*dmu_z'))
    bases = [('cc-pVDZ', 'cc-pvdz'), ('cc-pVTZ', 'cc-pvtz'), ('aug-cc-pVTZ', 'aug-cc-pvtz'),
             ('cc-pVQZ', 'cc-pvqz'), ('aug-cc-pVQZ', 'aug-cc-pvqz')]
    for beta in (3.0, 2.5, 2.0, 1.8):
        bases.append(('ETB beta=%.1f (F spdf, H spd)' % beta, etb_basis(beta)))
    for beta in (2.5, 2.0):
        bases.append(('ETB beta=%.1f (F spdfg, H spdf)' % beta, etb_basis(beta, 4, 3)))
    for label, basis in bases:
        mol = gto.M(atom=geom, basis=basis, verbose=0)
        dmu, nao, e = delta_mu(mol)
        print('%-28s %5d   % .6e   % .6e' % (label, nao, dmu, 2 * dmu), flush=True)

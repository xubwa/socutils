#!/usr/bin/env python

'''
Rare-gas atom in a cubic box of edge L: background Darwin term of periodic
SFX2C-1e and X2C-1e (spin-orbital, pyscf.pbc.x2c.x2c1e), the analogue of
30-box_jellium_darwin.py for several atoms and both schemes.

    PC 1st = Tr D_NR (h - h_NR)(box) - [same](isolated atom)
    d      = [E - E_NR](box) - [E - E_NR](isolated atom)        (SCF)
    exch   = -(2 pi/3) Delta[sum_p <p|r^2|p> - sum_pq |<p|r|q>|^2]  (p, q
             occupied spin orbitals of the same spin: the relativistic change
             of the nonrelativistic finite-size term of the Ewald exchange)

all in units of L^-3 c^-2.  Expected:  PC 1st -> -(pi/2) Z N (CBS) in both
schemes, and d = PC 1st + exch.

Usage:  python 47-box_darwin_so.py [c ...]      (env BOX_ATOMS="He Ne Ar")
'''

import os
import sys
import numpy
import scipy.linalg
from pyscf import lib, gto, scf
lib.einsum = numpy.einsum   # pyscf lib.einsum vs numpy 2.x
from pyscf.pbc import gto as pgto, scf as pscf
from pyscf.pbc.x2c import sfx2c1e as psfx, x2c1e as px2c

atoms = os.environ.get('BOX_ATOMS', 'He Ne Ar').split()
boxes = [12., 16., 20.]
cs = [float(x) for x in sys.argv[1:]] or [30., 137.036]
bas_name = {'He': 'cc-pvqz', 'Ne': 'cc-pvtz', 'Ar': 'cc-pvtz'}


def g2(x):
    return scipy.linalg.block_diag(x, x)


print('rare-gas atom in a cubic box, Gamma point, GDF, exxdiv=ewald;  units of L^-3 c^-2')
print('%3s %8s %5s %9s | %9s %9s %9s %9s | %9s %9s %9s %9s' % (
    '', 'c', 'L', 'PC CBS', 'SF 1st', 'SF exch', 'SF sum', 'SF d', 'SO 1st', 'SO exch', 'SO sum', 'SO d'))
for symb in atoms:
    basis = {symb: gto.uncontract(gto.load(bas_name[symb], symb))}
    for c in cs:
        lib.param.LIGHT_SPEED = c
        mol = gto.M(atom='%s 0 0 0' % symb, basis=basis, verbose=0)
        Z = N = mol.atom_charge(0)
        m_nr = scf.RHF(mol).run(conv_tol=1e-12)
        dm = m_nr.make_rdm1()
        dmg = g2(dm) * .5
        r2 = mol.intor('int1e_r2')
        rr = mol.intor('int1e_r')


        def fs(D, ghf=False):
            # sum_p <p|r^2|p> - sum_pq(same spin) |<p|r|q>|^2  (finite-size exchange)
            if ghf:
                return (numpy.einsum('ij,ji', D, g2(r2)) - sum(
                    numpy.einsum('ij,jk,kl,li', D, g2(x), D, g2(x)) for x in rr)).real
            return numpy.einsum('ij,ji', D, r2) - .5 * sum(numpy.einsum('ij,jk,kl,li', D, x, D, x) for x in rr)

        hnr_mol = m_nr.get_hcore()
        m_sf = scf.RHF(mol).sfx2c1e().run(conv_tol=1e-12)
        m_so = scf.GHF(mol).x2c1e().run(conv_tol=1e-12)
        mol_ref = {
            'sf': (m_sf.e_tot - m_nr.e_tot, numpy.einsum('ij,ji', dm, m_sf.get_hcore() - hnr_mol),
                   fs(m_sf.make_rdm1()) - fs(dm)),
            'so': (m_so.e_tot - m_nr.e_tot, numpy.einsum('ij,ji', dmg, m_so.get_hcore() - g2(hnr_mol)).real,
                   fs(m_so.make_rdm1(), True) - fs(dmg, True)),
        }
        for L in boxes:
            cell = pgto.Cell(atom='%s 0 0 0' % symb, a=numpy.eye(3) * L, basis=basis,
                             unit='B', verbose=0, precision=1e-12).build()
            k0 = numpy.zeros((1, 3))
            mf = pscf.RHF(cell).density_fit()
            mf.conv_tol = 1e-12
            e_nr = mf.kernel()
            h_nr = mf.get_hcore()
            out = {}
            for name in ('sf', 'so'):
                if name == 'sf':
                    mf = pscf.RHF(cell).density_fit().sfx2c1e()
                    h = psfx.SpinFreeX2CHelper(cell).get_hcore()
                    pc1 = numpy.einsum('ij,ji', dm, h - h_nr)
                else:
                    mf = pscf.KGHF(cell, k0).density_fit().x2c1e()
                    h = numpy.asarray(px2c.SpinOrbitalX2C1EHelper(cell).get_hcore(cell, k0))[0]
                    pc1 = numpy.einsum('ij,ji', dmg, h - g2(h_nr)).real
                mf.conv_tol = 1e-12
                e = mf.kernel()
                d_mol, pc_mol, dr2_mol = mol_ref[name]
                f = L**3 * c**2
                p1 = (pc1 - pc_mol) * f
                ex = -2 * numpy.pi / 3 * dr2_mol * c**2
                out[name] = (p1, ex, p1 + ex, (e - e_nr - d_mol) * f)
            print('%3s %8.3f %5.0f %9.4f | %9.4f %9.4f %9.4f %9.4f | %9.4f %9.4f %9.4f %9.4f' % (
                (symb, c, L, -numpy.pi / 2 * Z * N) + out['sf'] + out['so']), flush=True)

#!/usr/bin/env python

'''
Periodic sf-X2C-1e (pyscf.pbc.x2c): one He atom in a cubic box of edge L.

The Ewald nuclear potential (G = 0 removed) is the potential of the point
nuclei plus a uniform neutralizing background -rho = -Z/L^3, so near the atom
it carries the curvature of the background,  lap V = -4 pi rho  (checked
below: V_box - V_mol = v0 + k r^2, k = -(2 pi/3) Z/L^3).  X2C-1e picture-
changes this background together with the nuclei, while the background of
the electrons inside J is untransformed (in 4c the two backgrounds cancel).
The leftover is the Darwin term of the nuclear background,

    dE_PC = -(pi / 2c^2) rho N_e = -(pi / 2c^2) Z N_e / L^3,

a uniform shift of every orbital energy.  It is the G = 0 counterpart of the
multipole tails: nuclear background transformed, electronic background not.

The total d = [E_X2C - E_NR](box) - [E_X2C - E_NR](mol) also contains the
change of the ordinary (nonrelativistic) finite-size term of the Ewald
exchange, -(2 pi/3) <sum r^2> / L^3, through the relativistic contraction
of <sum r^2>.  That part is not a picture-change error (4c has the same term
with its own <r^2>) and vanishes with supercell / k-point convergence.

    d * L^3  =  -(pi/2c^2) Z N_e * (first order)  -  (2 pi/3) Delta<sum r^2>
'''

import sys
import numpy
from pyscf import lib, gto, scf
lib.einsum = numpy.einsum   # pyscf lib.einsum vs numpy 2.x
from pyscf.pbc import gto as pgto, scf as pscf, df as pdf
from pyscf.pbc.x2c import sfx2c1e as psfx

symb = 'He'
basis = {symb: gto.uncontract(gto.load('cc-pvqz', symb))}
Z = N = 2
boxes = [12., 16., 20.]
cs = [float(x) for x in sys.argv[1:]] or [10., 20., 137.036]

print('%s, uncontracted cc-pVQZ, Gamma point, GDF, exxdiv=ewald' % symb)
print('d = [E_X2C - E_NR](box) - [E_X2C - E_NR](mol);  units of L^-3 c^-2')
print('%8s %5s %10s %10s %10s %10s %10s %10s %10s' % (
    'c', 'L', 'k L^3', 'd', 'PC 1st', 'PC CBS', 'exch r^2', 'sum', 'NR box'))
for c in cs:
    lib.param.LIGHT_SPEED = c
    mol = gto.M(atom='%s 0 0 0' % symb, basis=basis, verbose=0)
    m_nr = scf.RHF(mol).run(conv_tol=1e-12)
    m_x = scf.RHF(mol).sfx2c1e().run(conv_tol=1e-12)
    dm = m_nr.make_rdm1()
    r2 = mol.intor('int1e_r2')
    s_m = mol.intor('int1e_ovlp')
    dr2 = numpy.einsum('ij,ji', m_x.make_rdm1() - dm, r2)
    dh_mol = numpy.einsum('ij,ji', dm, m_x.get_hcore() - m_nr.get_hcore())
    for L in boxes:
        cell = pgto.Cell(atom='%s 0 0 0' % symb, a=numpy.eye(3) * L, basis=basis,
                         unit='B', verbose=0, precision=1e-12).build()
        e = {}
        for name in ('nr', 'x2c'):
            mf = pscf.RHF(cell).density_fit()
            if name == 'x2c':
                mf = mf.sfx2c1e()
            mf.conv_tol = 1e-12
            e[name] = mf.kernel()
            if name == 'nr':
                h_nr = mf.get_hcore()
        h_x = psfx.SpinFreeX2CHelper(cell).get_hcore()
        # curvature of the cell nuclear potential relative to the molecule
        dv = pdf.DF(cell).get_nuc() - mol.intor('int1e_nuc')
        A = numpy.array([[numpy.vdot(s_m, s_m), numpy.vdot(s_m, r2)],
                         [numpy.vdot(r2, s_m), numpy.vdot(r2, r2)]])
        k = numpy.linalg.solve(A, [numpy.vdot(s_m, dv), numpy.vdot(r2, dv)])[1]
        d = e['x2c'] - e['nr'] - (m_x.e_tot - m_nr.e_tot)
        pc1 = numpy.einsum('ij,ji', dm, h_x - h_nr) - dh_mol
        f = L**3 * c**2
        exch = -2 * numpy.pi / 3 * dr2
        print('%8.3f %5.0f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f' % (
            c, L, k * L**3, d * f, pc1 * f, -numpy.pi / 2 * Z * N, exch * c**2,
            pc1 * f + exch * c**2, (e['nr'] - m_nr.e_tot) * L**3), flush=True)
print('k L^3 reference (jellium): -(2 pi/3) Z = %.4f;  NR box reference -(2 pi/3)<sum r^2>' % (-2 * numpy.pi / 3 * Z))

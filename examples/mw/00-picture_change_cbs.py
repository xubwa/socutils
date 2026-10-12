#!/usr/bin/env python

'''
Complete-basis-set (multiwavelet) c^-4 coefficient of the sf-X2C-1e
picture-change error of the lowest multipole of a molecule, i.e. the
coefficient of the spurious R^-(l+1) tail (a4 at fixed X, a4_X the '1e'
decoupling response) -Z_B dM / R^(l+1) caused by a far
charge Z_B on the +z axis.  See socutils/mw/picture_change.py.

    He2 (+-1.5 A):  O = r^2 P2 about the midpoint
    HF  (0.917 A):  O = z about the origin (F)

Usage:  python 00-picture_change_cbs.py he2|hf [prec1,prec2,...] [zeta]
'''

import sys
import time
import numpy
from pyscf import gto
from socutils import mw

system = sys.argv[1] if len(sys.argv) > 1 else 'he2'
precs = [float(p) for p in (sys.argv[2] if len(sys.argv) > 2 else '1e-4,1e-5').split(',')]
zeta = float(sys.argv[3]) if len(sys.argv) > 3 else 1e4

if system == 'he2':
    mol = gto.M(atom='He 0 0 -1.5; He 0 0 1.5', basis='cc-pvdz', verbose=0)
    op = mw.Multipole.quadrupole_zz()
else:
    mol = gto.M(atom='F 0 0 0; H 0 0 0.917', basis='cc-pvdz', verbose=0)
    op = mw.Multipole.dipole_z()

t0 = time.time()
phis = None
for prec in precs:
    ctx = mw.MWContext(mol, prec=prec, zeta=zeta)
    print('== %s  prec %.0e  order %d  box +-%d  zeta %g  int rho_nuc %.10f' % (
        system, prec, ctx.order, ctx.box, zeta, ctx.rho_nuc.integrate()), flush=True)
    if phis is None:
        phis, mf_gto = mw.project_mos(ctx, mol)
        print('  guess: GTO RHF/cc-pvdz (point nuclei) E_elec %.10f' % (mf_gto.e_tot - mf_gto.energy_nuc()))
    hf = mw.RHF(ctx, phis)
    hf.kernel()
    phis = hf.phis
    hf.save('orb_%s_%.0e' % (system, prec))
    r = mw.pc_coefficients(ctx, phis, op, gphis=hf.gphis())
    print('  RESULT prec %.0e  E_elec % .12f  <O> % .10e  a2 % .2e (exact 0)' % (
        prec, hf.e_elec, r['moment'], r['a2']))
    print('  RESULT prec %.0e  a4 % .8e = H1 % .8e + H2kin % .8e + W % .8e' % (
        prec, r['a4'], r['a4_H1'], r['a4_H2kin'], r['a4_W']))
    print('  RESULT prec %.0e  a4_X (1e decoupling response) % .8e  (check form % .8e)  a4 + a4_X % .8e' % (
        prec, r['a4_X'], r['a4_X_check'], r['a4'] + r['a4_X']))
    print('  per orbital <[T,[T,O]]>: %s   <W T^-1[O,T]>: %s  (%.0f s)' % (
        numpy.array2string(numpy.array([o['kin'] for o in r['orbitals']]), precision=6),
        numpy.array2string(numpy.array([o['w'] for o in r['orbitals']]), precision=6),
        time.time() - t0), flush=True)

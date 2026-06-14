#!/usr/bin/env python
'''
Constructing the X2C spin-orbit helper directly.

The .x2camf()/.x2cmp() shortcuts cover the common case.  For options the
shortcuts do not expose, build a SpinorX2CMPHelper (spinor basis) or a
SpinOrbitalX2CMPHelper (GHF basis) and assign it to with_x2c yourself.
'''
from pyscf import gto
from socutils.scf import spinor_hf, ghf
from socutils.somf.x2cmp import SpinorX2CMPHelper, SpinOrbitalX2CMPHelper

mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

# spinor basis
mf = spinor_hf.SCF(mol)
mf.with_x2c = SpinorX2CMPHelper(mol, x2cmp='x2camf',
                                with_gaunt=True, with_breit=True,
                                with_pcc=False, with_aoc=False)
print('E(spinor, helper) = %.10f' % mf.kernel())

# GHF (spin-orbital) basis
gmf = ghf.GHF(mol)
gmf.with_x2c = SpinOrbitalX2CMPHelper(mol, x2cmp='x2camf',
                                      with_gaunt=True, with_breit=True)
print('E(GHF, helper)    = %.10f' % gmf.kernel())

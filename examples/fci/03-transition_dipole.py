#!/usr/bin/env python

'''
Transition density matrices, transition dipole moments and oscillator
strengths between CASCI states.

addons.transition_dipole / addons.oscillator_strength work with any
fcisolver that provides trans_rdm1(ci_bra, ci_ket, ncas, nelecas), such
as zfci.FCISolver and zfci.SelectedCI.  All roots are obtained from a
single diagonalization, so the full set of transition moments between
any pair of states is available.

With spin-orbit coupling included (X2C), spin symmetry is no longer
exact: transitions to triplet-dominated states acquire a small
oscillator strength borrowed from the singlet manifold.  For transitions
involving degenerate states (Kramers multiplets), the individual
oscillator strengths depend on the arbitrary unitary within the
degenerate space; only the sum over a degenerate manifold is physical.
'''

import numpy
from pyscf import gto, scf
from pyscf.data import nist
from socutils.mcscf import zcasci
from socutils.fci import zfci, addons

mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis='sto-3g', verbose=0)
mf = scf.X2C(mol).run()

mc = zcasci.CASCI(mf, 8, 4)
mc.fcisolver = zfci.FCISolver(mol)
mc.fcisolver.nroots = 8
mc.kernel()
energies = numpy.asarray(mc.fcisolver.eci).real

#
# Permanent dipole moment of the ground state.  The nuclear dipole
# vanishes at the charge center (the default origin), so the electronic
# part is already the total dipole moment of the neutral molecule.
#
dip = addons.transition_dipole(mc, 0, 0).real
print('Ground state dipole moment (Debye): %.4f %.4f %.4f'
      % tuple(dip * nist.AU2DEBYE))

#
# Excitation energies and oscillator strengths from the ground state.
# States 1-3 are the triplet-like Kramers multiplet: their oscillator
# strength is tiny, induced purely by spin-orbit coupling.  State 4 is
# the dipole-allowed singlet-like state.
#
print('state   Eex/eV    f(length gauge)')
for i in range(1, 8):
    f, f_xyz = addons.oscillator_strength(mc, 0, i)
    de = (energies[i] - energies[0]) * nist.HARTREE2EV
    print('%4d   %8.4f   %.6e' % (i, de, f))

#
# Transition dipole moments between any two states; for i == j the core
# contribution is included automatically.
#
t_dip = addons.transition_dipole(mc, 0, 4)
print('|<0|r|4>| = %.6f a.u.' % numpy.linalg.norm(t_dip))

#
# The transition density matrix itself (AO spinor basis) is available,
# e.g. for plotting; it contracts with one-electron operators O_ao as
# <i|O|j> = einsum('uv,vu->', O_ao, dm).
#
dm_t = addons.trans_rdm1_ao(mc, 0, 4)
print('norm of AO transition density:', numpy.linalg.norm(dm_t).round(6))

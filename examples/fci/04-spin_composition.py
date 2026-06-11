#!/usr/bin/env python

'''
Angular momentum analysis of relativistic CI states.

With spin-orbit coupling, S and L are no longer good quantum numbers.
The expectation values <S^2>, <L^2> and <J^2> are evaluated from the
active space 1- and 2-RDMs:

    <(sum_i o_i)^2> = sum_pq (o o)_pq <p^+ q>
                    + sum_pqrs o_pq o_rs <p^+ r^+ s q>

with the one-electron matrices o taken as the spin (sigma/2), orbital
angular momentum (-i r x nabla) or total angular momentum matrices in
the spinor MO basis.

* <S^2> quantifies the singlet/triplet composition of a SOC-mixed state:
  it interpolates between 0 (pure singlet) and 2 (pure triplet).
* <J^2> assigns the j quantum number of atomic states, J^2 = j(j+1).
'''

import numpy
from pyscf import gto, scf
from pyscf.data import nist
from socutils.mcscf import zcasci
from socutils.fci import zfci, addons

#
# 1. Molecular states: singlet/triplet composition of water X2C-CASCI
#    states.  For light elements the SOC mixing is tiny, so <S^2> stays
#    very close to the pure-spin values 0 and 2.
#
mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis='sto-3g', verbose=0)
mf = scf.X2C(mol).run()
mc = zcasci.CASCI(mf, 8, 4)
mc.fcisolver = zfci.FCISolver(mol)
mc.fcisolver.nroots = 5
mc.kernel()
e = numpy.asarray(mc.fcisolver.eci).real
print('water X2C-CASCI states:')
for i in range(5):
    ss, mult = addons.spin_square(mc, i)
    print('  state %d  Eex = %7.4f eV   <S^2> = %.8f   2S+1 = %.6f'
          % (i, (e[i]-e[0])*nist.HARTREE2EV, ss, mult))

#
# 2. Atomic states: j-classification of the F 2p^5 multiplet.  The
#    orbitals are taken from closed-shell F- to avoid symmetry breaking;
#    the CASCI then puts 5 electrons in the perfectly degenerate 2p
#    spinors.  J^2 = 3.75 identifies j = 3/2 and J^2 = 0.75 j = 1/2;
#    the splitting is the 2P spin-orbit splitting of fluorine.
#
mol = gto.M(atom='F 0 0 0', basis='unc-sto-3g', charge=-1, verbose=0)
mf = scf.X2C(mol).run()
mc = zcasci.CASCI(mf, 6, 5, ncore=4)
mc.fcisolver = zfci.FCISolver(mol)
mc.fcisolver.nroots = 6
mc.kernel()
e = numpy.asarray(mc.fcisolver.eci).real
au2cm = nist.HARTREE2WAVENUMBER
print('F 2p^5 multiplet (exp. splitting 404 cm-1):')
for i in range(6):
    j2 = addons.angular_momentum_square(mc, i, kind='total')
    l2 = addons.angular_momentum_square(mc, i, kind='orb')
    ss = addons.angular_momentum_square(mc, i, kind='spin')
    j = 0.5*(numpy.sqrt(1 + 4*j2) - 1)
    print('  state %d  E-E0 = %7.2f cm-1   J^2 = %.4f (j = %.1f)'
          '  L^2 = %.4f  S^2 = %.4f' % (i, (e[i]-e[0])*au2cm, j2, j, l2, ss))

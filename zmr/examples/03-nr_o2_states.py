#!/usr/bin/env python
'''
Non-relativistic multi-state NEVPT2 with zmr.nr: the three lowest states of
O2 (X 3Sigma_g-, a 1Delta_g, b 1Sigma_g+).

State-averaged CASSCF(8e, 6o) over the four lowest Ms = 0 states (the Ms = 0
component of the triplet, both components of a 1Delta_g, and b 1Sigma_g+),
then spin-free PC- and UC-NEVPT2 with one effective Hamiltonian for all four
states.  The perturber functions are spin-free, so the triplet could equally
be given as its Ms = 1 component, and states of different spin share a model
space as long as they have the same (n_alpha, n_beta).

Usage:  python 03-nr_o2_states.py [basis]

Experiment (T_e): a 1Delta_g 0.98 eV, b 1Sigma_g+ 1.64 eV.
'''
import sys
import time

import numpy
from pyscf import gto, mcscf, scf

from socutils.zmr.interface import pyscf_nr
from socutils.zmr.nr import PCNEVPT2, UCNEVPT2

HARTREE2EV = 27.211386245988
basis = sys.argv[1] if len(sys.argv) > 1 else 'cc-pvtz'

mol = gto.M(atom='O 0 0 0; O 0 0 1.2075', basis=basis, spin=2, verbose=0)
mf = scf.ROHF(mol).run(conv_tol=1e-11)
mc = mcscf.CASSCF(mf, 6, (4, 4))             # Ms = 0 sector of the 2p valence space
mc = mc.state_average_([0.25] * 4)
mc.conv_tol = 1e-10
mc.conv_tol_grad = 1e-6
mc.kernel()
print('O2 / %s, %d basis functions, SA-CASSCF(8,6) states (Eh): %s'
      % (basis, mol.nao, numpy.array2string(numpy.asarray(mc.e_states), precision=8)))

# real spatial integrals, 1s frozen; the CASSCF CI vectors are the model states
eris, nelec = pyscf_nr.nr_from_casci(mc, nfrozen=2)
civecs = pyscf_nr.nr_civecs(mc)


def report(label, e):
    ev = (e - e[0]) * HARTREE2EV
    print('%-12s E(X) = %.8f   a 1Delta_g %.3f, %.3f eV   b 1Sigma_g+ %.3f eV'
          % (label, e[0], ev[1], ev[2], ev[3]))


for name, cls in (('PC-NEVPT2', PCNEVPT2), ('UC-NEVPT2', UCNEVPT2)):
    t0 = time.time()
    pt = cls(eris, nelec, civecs=civecs, e0='average')
    e = pt.kernel()
    if name == 'PC-NEVPT2':
        report('CASSCF', numpy.linalg.eigvalsh(pt.hmodel))
    report(name, e)
    print('             second-order energy of X by class (Eh): %s   [%.1f s]'
          % (' '.join('%s %.6f' % (k, v[0, 0].real) for k, v in pt.h2_by_class.items()),
             time.time() - t0))

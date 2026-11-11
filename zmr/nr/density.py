'''
Unrelaxed (response) one-particle density of the spin-free PC-NEVPT2 (zmr.nr) for a
singlet model state, through the spinor code on the spin-orbital form of the same
Hamiltonian: for singlets the spin-free contracted functions and the spin-orbital ones
span the same first-order spaces (zmr/tests/test_nr_nevpt2.py), so the second-order
energy and its derivatives coincide.  The spin-orbital density (zmr.uc.density.make_rdm1,
spin-orbitals p = 2P + sigma) is summed over the spin.

    ref, d2 = make_rdm1(mc, cderi)     spatial, D[P, Q] = sum_sigma <P_sigma^+ Q_sigma>
    dE2 / dlambda = sum O[P, Q] d2[P, Q]   for h -> h + lambda O at fixed orbitals and CI
'''

import numpy

from socutils.zmr.interface import pyscf_nr
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.uc import density


def spin_sum(d):
    '''Spatial density from a spin-orbital one (p = 2P + sigma).'''
    return d[0::2, 0::2] + d[1::2, 1::2]


def make_rdm1(mc, cderi, max_memory=4000, timings=None):
    '''(D_ref, D_pt2, pt): the reference and unrelaxed second-order 1-RDMs of the singlet
    ground state of the PySCF CASSCF/CASCI mc, in its orbitals (all orbitals correlated),
    and the spin-orbital PC-NEVPT2 they come from.  cderi: three-index vectors of the
    AO integrals (zmr.interface.cholesky).'''
    if mc.mol.spin != 0:
        raise NotImplementedError('the spin-orbital route equals the spin-free PC-NEVPT2 '
                                  'for singlets only')
    eris, nelec = pyscf_nr.from_casci(mc, cderi=cderi)
    pt = PCNEVPT2(eris, nelec, nstate=1)
    pt.verbose = 0
    pt.max_memory = max(pt.max_memory, max_memory)
    pt.kernel()
    ref, d2 = density.make_rdm1(pt, max_memory=max_memory, timings=timings)
    return spin_sum(ref), spin_sum(d2), pt

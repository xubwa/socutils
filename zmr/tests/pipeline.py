'''Common setup: model states, averaged Fock and quasi-canonical orbitals.'''
import numpy

from socutils.zmr import fock
from socutils.zmr.active.sectors import ActiveSpace


class Setup:
    '''Quasi-canonical Hamiltonian and model states for a given Hamiltonian.

    Model states are the nstate lowest CASCI states (or the given CI vectors);
    the Fock matrix uses their weighted-average 1-RDM.'''

    def __init__(self, eris, nelec, nstate=1, weights=None, civecs=None):
        act = ActiveSpace.from_eris(eris)
        if civecs is None:
            _, civecs = act.solve(nelec, nstate)
        civecs = numpy.asarray(civecs)
        self.nelec = nelec
        self.civecs = civecs
        self.weights = weights
        gamma = act.averaged_rdm1(civecs, nelec, weights)
        qc = fock.quasi_canonicalize(eris, gamma)
        self.gamma = gamma
        self.eris = qc.eris
        self.fock = qc.fock
        self.eps_core = qc.eps_core
        self.eps_virt = qc.eps_virt

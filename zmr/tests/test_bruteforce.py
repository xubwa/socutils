'''Internal consistency of the brute-force UC-NEVPT2 reference.'''
import numpy

from socutils.zmr.reference.nevpt2_bruteforce import BruteForceNEVPT2
from socutils.zmr.tests._helpers import model_eris
from socutils.zmr.tests.pipeline import Setup


def test_dyall_hamiltonian_equals_h_on_cas_space():
    eris = model_eris(3, 4, 3, seed=1)
    s = Setup(eris, 2, nstate=3)
    bf = BruteForceNEVPT2(s.eris, s.eps_core, s.eps_virt, 2, s.civecs)
    heff, h2 = bf.kernel()
    # V has no CAS-CAS block, and the model space Hamiltonian is diagonal
    assert bf.cas_leak < 1e-12
    assert abs(bf.hmodel - numpy.diag(bf.e_cas)).max() < 1e-12
    # second order: Hermitian, and the ground state is lowered
    assert abs(bf.h2 - bf.h2.conj().T).max() < 1e-12
    assert bf.h2[0, 0].real < 0
    for a in h2.values():
        assert numpy.all(a.diagonal().real <= 1e-14)


def test_e0_options_agree_for_one_state():
    eris = model_eris(2, 4, 2, seed=2)
    s = Setup(eris, 2, nstate=1)
    e = []
    for e0 in ('state', 'average'):
        bf = BruteForceNEVPT2(s.eris, s.eps_core, s.eps_virt, 2, s.civecs, e0=e0)
        bf.kernel()
        e.append(bf.energies[0])
    assert abs(e[0] - e[1]) < 1e-12

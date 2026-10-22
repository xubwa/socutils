'''Exact (spectral) UC-NEVPT2 engine vs the brute-force reference, class by
class (validation item 1).'''
import numpy
import pytest

from socutils.zmr.interface import pyscf_nr
from socutils.zmr.reference.nevpt2_bruteforce import BruteForceNEVPT2
from socutils.zmr.tests import systems
from socutils.zmr.tests._helpers import model_eris
from socutils.zmr.tests.pipeline import Setup
from socutils.zmr.uc import classes
from socutils.zmr.uc.nevpt2 import UCNEVPT2

TOL = 1e-10


def _compare(eris, nelec, nstate, e0, weights=None):
    uc = UCNEVPT2(eris, nelec, nstate=nstate, e0=e0, weights=weights)
    uc.kernel()
    s = Setup(eris, nelec, civecs=uc.civecs, weights=weights)
    bf = BruteForceNEVPT2(s.eris, s.eps_core, s.eps_virt, nelec, uc.civecs,
                          e0=e0, weights=weights)
    bf.kernel()
    assert bf.cas_leak < 1e-12
    for name in bf.h2_by_class:
        assert abs(uc.h2_by_class[name] - bf.h2_by_class[name]).max() < TOL, name
    assert abs(uc.heff - bf.heff).max() < TOL
    assert abs(uc.energies - bf.energies).max() < TOL
    return uc, bf


@pytest.mark.parametrize('nelec', [1, 2, 3])
@pytest.mark.parametrize('nstate', [1, 3])
@pytest.mark.parametrize('e0', ['state', 'average'])
def test_random_complex_hamiltonian(nelec, nstate, e0):
    eris = model_eris(3, 4, 3, seed=10 * nelec + nstate)
    uc, _ = _compare(eris, nelec, nstate, e0)
    # every class contributes and the second-order matrix is Hermitian
    nonzero = ['ccvv', 'ccav', 'cavv', 'cava', 'aaav', 'caaa']
    nonzero += ['aavv'] if nelec >= 2 else []      # needs two active electrons
    nonzero += ['ccaa'] if nelec <= 2 else []      # needs two active holes (nact = 4)
    for name in nonzero:
        assert abs(uc.h2_by_class[name]).max() > 1e-8, name
    assert abs(uc.h2 - uc.h2.conj().T).max() < 1e-12
    assert abs(eris.h1.imag).max() > 1e-3          # genuinely complex integrals
    if nstate > 1:
        assert abs(uc.heff.imag).max() > 1e-8


def test_unequal_weights():
    eris = model_eris(2, 4, 3, seed=5)
    _compare(eris, 2, 3, 'average', weights=[0.5, 0.3, 0.2])


@pytest.mark.parametrize('nstate', [1, 3])
def test_h2o_sto3g_cas22(nstate):
    mc = systems.h2o_casci(2, 2)
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=1)
    _compare(eris, nelec, nstate, 'state')


@pytest.mark.slow
def test_h2o_631g_cas44():
    mc = systems.h2o_casci(4, 4, basis='6-31g')
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=1)
    uc, _ = _compare(eris, nelec, 1, 'state')
    for name in [c.name for c in classes.ALL_CLASSES]:
        assert abs(uc.h2_by_class[name]).max() > 1e-4, name


def test_target_compression():
    '''ccvv (O = 1): the model space is H_act invariant, so its first-order
    space reduces exactly to the model states; the span of ccav is not
    invariant and the whole sector is used.'''
    from socutils.zmr.active.sectors import ActiveSpace
    from socutils.zmr.uc import engine
    from socutils.zmr.uc.nevpt2 import invariant_span
    eris = model_eris(2, 6, 2, seed=12)
    act = ActiveSpace.from_eris(eris)
    _, c = act.solve(3, 3)
    phi = engine.class_functions(classes.CLASS_BY_NAME['ccvv'], act, c, 3)
    basis, mu = invariant_span(act, 3, phi)
    assert basis.shape[1] == 3
    assert abs(numpy.sort(mu) - act.sector(3).lam[:3]).max() < 1e-12
    phi = engine.class_functions(classes.CLASS_BY_NAME['ccav'], act, c, 3)
    assert invariant_span(act, 4, phi) is None

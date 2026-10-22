'''Multi-state properties: invariance under unitary rotations of the model
space (conventions.md section 9).'''
import numpy
import pytest
from scipy.stats import unitary_group

from socutils.zmr.interface import pyscf_nr
from socutils.zmr.tests import systems
from socutils.zmr.tests._helpers import model_eris
from socutils.zmr.uc.nevpt2 import UCNEVPT2


def _rotated_pair(eris, nelec, nstate, e0, method='exact', seed=0):
    ref = UCNEVPT2(eris, nelec, nstate=nstate, e0=e0, method=method, nquad=48)
    ref.kernel()
    u = unitary_group.rvs(nstate, random_state=seed)
    rot = UCNEVPT2(eris, nelec, civecs=ref.civecs @ u, e0=e0, method=method, nquad=48)
    rot.kernel()
    return ref, rot, u


@pytest.mark.parametrize('method', ['exact', 'laplace'])
def test_rotation_invariance_with_average_e0(method):
    eris = model_eris(3, 4, 3, seed=31)
    ref, rot, u = _rotated_pair(eris, 2, 4, 'average', method)
    assert abs(rot.energies - ref.energies).max() < 1e-10
    assert abs(rot.heff - u.conj().T @ ref.heff @ u).max() < 1e-10
    # the model-space Hamiltonian is no longer diagonal after the rotation
    assert abs(rot.hmodel - numpy.diag(rot.hmodel.diagonal())).max() > 1e-3


def test_rotation_invariance_h2o():
    mc = systems.h2o_casci(4, 4, basis='6-31g')
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=1)
    ref, rot, _ = _rotated_pair(eris, nelec, 3, 'average', seed=3)
    assert abs(rot.energies - ref.energies).max() < 1e-10


def test_state_specific_e0_is_not_rotation_invariant():
    '''Per-state E0 uses the model-state energies in the resolvents, so a
    rotation of the model space changes the result by construction.'''
    eris = model_eris(3, 4, 3, seed=31)
    ref, rot, _ = _rotated_pair(eris, 2, 4, 'state')
    assert abs(rot.energies - ref.energies).max() > 1e-6


def test_unequal_weights_are_not_rotation_invariant():
    eris = model_eris(3, 4, 3, seed=31)
    w = [0.4, 0.3, 0.2, 0.1]
    ref = UCNEVPT2(eris, 2, nstate=4, e0='average', weights=w)
    ref.kernel()
    u = unitary_group.rvs(4, random_state=1)
    rot = UCNEVPT2(eris, 2, civecs=ref.civecs @ u, e0='average', weights=w)
    rot.kernel()
    assert abs(rot.energies - ref.energies).max() > 1e-6


def test_single_state_limit():
    '''For one state both E0 choices coincide, and the energy is E_CAS + E2.'''
    eris = model_eris(3, 4, 3, seed=32)
    a = UCNEVPT2(eris, 2, e0='state')
    b = UCNEVPT2(eris, 2, e0='average')
    a.kernel()
    b.kernel()
    assert abs(a.energies - b.energies).max() < 1e-12
    assert abs(a.energies[0] - (a.hmodel[0, 0] + a.h2[0, 0]).real) < 1e-12

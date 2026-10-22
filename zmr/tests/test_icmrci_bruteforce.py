'''Brute-force internally contracted MRCI: exact limits and bounds.'''
import numpy
import pytest

from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.interface import pyscf_nr
from socutils.zmr.reference import icmrci_bruteforce, mrci_bruteforce
from socutils.zmr.tests import systems
from socutils.zmr.tests._helpers import model_eris


def _cas(eris, nelec, nstate):
    act = ActiveSpace.from_eris(eris)
    return act.solve(nelec, nstate)


@pytest.mark.parametrize('seed', [1, 2])
def test_bounds(seed):
    '''E(uncontracted MRCISD) <= E(ic, relaxed reference) <= E(ic, fixed
    states) <= E(CAS), root by root for the lowest state.'''
    eris = model_eris(2, 4, 3, seed=seed)
    nelec = 2
    e_cas, c = _cas(eris, nelec, 1)
    e_unc = mrci_bruteforce.kernel(eris, nelec)[0]
    relaxed = icmrci_bruteforce.kernel(eris, nelec, c)
    fixed = icmrci_bruteforce.kernel(eris, nelec, c, internal='states')
    assert e_unc[0] <= relaxed.e[0] + 1e-10
    assert relaxed.e[0] <= fixed.e[0] + 1e-10
    assert fixed.e[0] < e_cas[0] - 1e-6
    # a genuine contraction: fewer functions than determinants, and a higher energy
    assert relaxed.basis.shape[1] < relaxed.ndet
    assert relaxed.e[0] - e_unc[0] > 1e-9
    b = relaxed.basis
    assert abs(b.conj().T @ b - numpy.eye(b.shape[1])).max() < 1e-10


def test_all_cas_states_as_references_is_uncontracted():
    '''tau |Psi_J> for J spanning the CAS space spans tau times every CAS
    determinant: the union contraction over all CAS states is no contraction.'''
    eris = model_eris(2, 3, 2, seed=4)
    nelec = 2
    act = ActiveSpace.from_eris(eris)
    e_cas, c = act.solve(nelec, act.dim(nelec))
    e_unc = mrci_bruteforce.kernel(eris, nelec, nroots=3)[0]
    res = icmrci_bruteforce.kernel(eris, nelec, c, nroots=3)
    assert res.basis.shape[1] == res.ndet
    assert abs(res.e - e_unc).max() < 1e-10


def test_single_determinant_reference_is_cisd():
    mf = systems.h2o_rhf('sto-3g')
    nocc = mf.mol.nelectron // 2
    eris = pyscf_nr.from_mo_coeff(mf.mol, mf.get_hcore(), mf.mo_coeff, 1, nocc - 1, 0)
    e_unc = mrci_bruteforce.kernel(eris, 0)[0]
    res = icmrci_bruteforce.kernel(eris, 0, numpy.ones((1, 1)))
    assert abs(res.e[0] - e_unc[0]) < 1e-10
    assert res.basis.shape[1] == res.ndet


def test_two_states_union():
    '''Two reference states: both roots are bounded by the uncontracted
    ones, and the classes of the contracted functions are those of
    PC-NEVPT2.'''
    eris = model_eris(2, 4, 2, seed=5)
    nelec = 3
    e_cas, c = _cas(eris, nelec, 2)
    e_unc = mrci_bruteforce.kernel(eris, nelec, nroots=2)[0]
    res = icmrci_bruteforce.kernel(eris, nelec, c, nroots=2)
    assert numpy.all(e_unc <= res.e + 1e-10)
    assert numpy.all(res.e < e_cas)
    assert set(res.nbasis_by_class) <= {'cas', 'ccvv', 'ccav', 'cavv', 'ccaa', 'aavv',
                                        'cava', 'caaa', 'aaav'}
    assert res.nbasis_by_class['cas'] == ActiveSpace.from_eris(eris).dim(nelec)

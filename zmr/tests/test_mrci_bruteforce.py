'''Brute-force determinant MRCISD: exact limits.'''
import numpy
import pytest
from pyscf import ci
from socutils.fci import zfci

from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.interface import pyscf_nr
from socutils.zmr.reference import mrci_bruteforce
from socutils.zmr.tests import systems
from socutils.zmr.tests._helpers import model_eris


def test_mrcisd_equals_fci_with_two_core_two_virtual():
    '''With two core and two virtual spin-orbitals every determinant has at
    most two holes and two particles: MRCISD is FCI.'''
    eris = model_eris(2, 4, 2, seed=1)
    nelec = 2
    e, c, strs = mrci_bruteforce.kernel(eris, nelec, nroots=4)
    e_fci, _ = zfci.kernel(eris.h1, eris.eri_block('pppp'), eris.nmo, eris.spaces.ncore + nelec,
                           ecore=eris.ecore, nroots=4)
    assert abs(e - e_fci).max() < 1e-10
    assert len(strs) == 70                      # C(8, 4)


def test_mrcisd_is_variational_and_below_cas():
    eris = model_eris(3, 4, 3, seed=2)
    e, _, strs = mrci_bruteforce.kernel(eris, 2, nroots=3)
    e_cas, _ = ActiveSpace.from_eris(eris).solve(2, 3)
    e_fci, _ = zfci.kernel(eris.h1, eris.eri_block('pppp'), eris.nmo, 5, ecore=eris.ecore, nroots=3)
    assert numpy.all(e < e_cas)
    assert numpy.all(e >= e_fci - 1e-10)
    assert len(strs) < 252                      # a proper subspace of FCI, C(10, 5) = 252


def test_block_restriction():
    '''Only the internal block: the CAS-CI energies.'''
    eris = model_eris(3, 4, 3, seed=3)
    e, _, _ = mrci_bruteforce.kernel(eris, 2, nroots=3, blocks=[(0, 0)])
    e_cas, _ = ActiveSpace.from_eris(eris).solve(2, 3)
    assert abs(e - e_cas).max() < 1e-10


@pytest.mark.parametrize('nfrozen', [0, 1])
def test_empty_active_space_is_cisd(nfrozen):
    mf = systems.h2o_rhf('sto-3g')
    nocc = mf.mol.nelectron // 2
    eris = pyscf_nr.from_mo_coeff(mf.mol, mf.get_hcore(), mf.mo_coeff, nfrozen, nocc - nfrozen, 0)
    e, _, _ = mrci_bruteforce.kernel(eris, 0)
    myci = ci.CISD(mf, frozen=nfrozen or None)
    myci.verbose = 0
    myci.conv_tol = 1e-12
    myci.kernel()
    assert abs(e[0] - myci.e_tot) < 1e-8

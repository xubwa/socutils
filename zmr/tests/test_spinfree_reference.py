'''UC-NEVPT2 vs the independent spatial-orbital brute-force reference
(full FCI space, pyscf.fci.direct_spin1), class by class.  This is the
external check that UC-NEVPT2 otherwise lacks: the two implementations share
only the PySCF integrals.'''
import pytest

from socutils.zmr.interface import pyscf_nr
from socutils.zmr.reference.nevpt2_spinfree import spinfree_nevpt2
from socutils.zmr.tests import systems
from socutils.zmr.uc.nevpt2 import UCNEVPT2


@pytest.mark.parametrize('basis, ncas, nelecas, nfrozen', [
    ('sto-3g', 2, 2, 1),
    ('sto-3g', 3, 4, 0),
    ('6-31g', 4, 4, 1),
])
def test_uc_nevpt2_matches_spinfree_bruteforce(basis, ncas, nelecas, nfrozen):
    mc = systems.h2o_casci(ncas, nelecas, basis=basis)
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=nfrozen)
    uc = UCNEVPT2(eris, nelec)
    uc.kernel()
    e_cas, e2, leak = spinfree_nevpt2(mc.mol, mc.get_hcore(), mc.mo_coeff, nfrozen,
                                      mc.ncore - nfrozen, mc.ncas, mc.nelecas)
    assert leak < 1e-12
    assert abs(e_cas - uc.hmodel[0, 0].real) < 1e-10
    for name, value in e2.items():
        assert abs(value - uc.h2_by_class[name][0, 0].real) < 1e-10, name

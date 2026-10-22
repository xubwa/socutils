'''Comparisons with PySCF (validation item 2).'''
import io
import re

import numpy
import pytest
from pyscf import mp
from pyscf.mrpt import nevpt2

from socutils.zmr.interface import pyscf_nr
from socutils.zmr.tests import systems
from socutils.zmr.uc.nevpt2 import UCNEVPT2

# PySCF SC-NEVPT2 subspace label -> perturber class
SC_LABEL = {'Sr': 'aaav', 'Si': 'caaa', 'Sijrs': 'ccvv', 'Sijr': 'ccav',
            'Srsi': 'cavv', 'Srs': 'aavv', 'Sij': 'ccaa', 'Sir': 'cava'}


def _sc_nevpt2_by_class(mc):
    '''Per-subspace SC-NEVPT2 energies parsed from the PySCF log.'''
    nev = nevpt2.NEVPT(mc)
    nev.stdout = io.StringIO()
    nev.verbose = 4
    nev.kernel()
    out = {}
    for line in nev.stdout.getvalue().splitlines():
        m = re.match(r"\s*(S\w+)\s+\([-+0-9]+\)'?\s*,\s*E = (\S+)", line)
        if m:
            out[SC_LABEL[m.group(1)]] = float(m.group(2))
    assert len(out) == 8
    return out, nev


@pytest.fixture(scope='module')
def h2o_631g():
    mc = systems.h2o_casci(4, 4, basis='6-31g')
    sc, nev = _sc_nevpt2_by_class(mc)
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=0)
    uc = UCNEVPT2(eris, nelec)
    uc.kernel()
    return sc, uc


def test_ccvv_equals_pyscf_sijrs(h2o_631g):
    '''The ccvv class is the same in all NEVPT2 variants.'''
    sc, uc = h2o_631g
    assert abs(uc.h2_by_class['ccvv'][0, 0].real - sc['ccvv']) < 1e-10


def test_uncontracted_bounded_by_strongly_contracted(h2o_631g):
    '''The SC perturbers of each class span a subspace of the UC first-order
    space and the classes are H_D-orthogonal, so by the Hylleraas principle
    E_UC(class) <= E_SC(class) for every class.'''
    sc, uc = h2o_631g
    for name, e_sc in sc.items():
        e_uc = uc.h2_by_class[name][0, 0].real
        assert e_uc <= e_sc + 1e-10, name
        assert e_uc < 0
    # not trivially equal: contraction matters for the classes with active indices
    assert abs(uc.h2_by_class['cava'][0, 0].real - sc['cava']) > 1e-5


@pytest.mark.parametrize('nfrozen', [0, 1])
def test_empty_active_space_is_mp2(nfrozen):
    mf = systems.h2o_rhf('6-31g')
    nocc = mf.mol.nelectron // 2
    eris = pyscf_nr.from_mo_coeff(mf.mol, mf.get_hcore(), mf.mo_coeff, nfrozen,
                                 nocc - nfrozen, 0)
    uc = UCNEVPT2(eris, 0)
    uc.kernel()
    e_mp2 = mp.MP2(mf, frozen=nfrozen or None).run(verbose=0).e_corr
    assert abs(uc.hmodel[0, 0].real - mf.e_tot) < 1e-10
    assert abs(uc.h2[0, 0].real - e_mp2) < 1e-10
    # only ccvv survives; cava keeps the f_ai^2 singles term, which vanishes
    # up to the SCF convergence (Brillouin theorem)
    for name, a in uc.h2_by_class.items():
        if name != 'ccvv':
            assert abs(a).max() < 1e-14, name

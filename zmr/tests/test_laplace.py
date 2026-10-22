'''Laplace-transformed UC-NEVPT2 converges to the exact result as the
quadrature is refined (validation item 1, Laplace part).'''
import numpy
import pytest

from socutils.zmr.interface import pyscf_nr
from socutils.zmr.tests import systems
from socutils.zmr.tests._helpers import model_eris
from socutils.zmr.uc import classes, laplace, laplace_engine
from socutils.zmr.uc.nevpt2 import UCNEVPT2


@pytest.mark.parametrize('ratio', [2.0, 10.0, 100.0])
def test_quadrature_converges(ratio):
    x = numpy.geomspace(0.7, 0.7 * ratio, 5000)
    errors = []
    for n in (10, 20, 30, 40, 60):
        q = laplace.quadrature(0.7, 0.7 * ratio, n)
        err = abs(q(x) * x - 1).max()
        # the reported estimate is reliable
        assert err <= 2 * q.max_rel_error + 1e-15
        errors.append(err)
    assert all(b <= a * 1.01 + 1e-15 for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 1e-11


def test_quadrature_rejects_nonpositive_range():
    with pytest.raises(ValueError):
        laplace.quadrature(-0.1, 1.0, 20)


@pytest.mark.parametrize('e0', ['state', 'average'])
def test_laplace_converges_to_exact(e0):
    eris = model_eris(3, 4, 3, seed=7)
    ex = UCNEVPT2(eris, 2, nstate=3, e0=e0)
    ex.kernel()
    errors = []
    for n in (8, 16, 24, 32, 48):
        lp = UCNEVPT2(eris, 2, nstate=3, e0=e0, method='laplace', nquad=n)
        lp.kernel()
        errors.append(max(abs(lp.h2_by_class[k] - ex.h2_by_class[k]).max()
                          for k in ex.h2_by_class))
    assert all(b < a for a, b in zip(errors[:3], errors[1:4]))
    assert errors[-1] < 1e-12


def test_laplace_h2o_all_electron():
    '''No frozen core: the core 1s widens the denominator range.'''
    mc = systems.h2o_casci(4, 4, basis='6-31g')
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=0)
    ex = UCNEVPT2(eris, nelec, nstate=2)
    ex.kernel()
    lp = UCNEVPT2(eris, nelec, nstate=2, method='laplace', nquad=60)
    lp.kernel()
    assert abs(lp.heff - ex.heff).max() < 1e-10
    assert abs(lp.energies - ex.energies).max() < 1e-10


def test_mixed_methods():
    eris = model_eris(3, 4, 3, seed=8)
    ex = UCNEVPT2(eris, 2, nstate=2)
    ex.kernel()
    mixed = UCNEVPT2(eris, 2, nstate=2, method={'ccav': 'laplace', 'cavv': 'laplace'}, nquad=48)
    mixed.kernel()
    assert abs(mixed.heff - ex.heff).max() < 1e-12


def test_intruder_is_reported():
    eris = model_eris(2, 4, 2, seed=9)
    uc = UCNEVPT2(eris, 2)
    uc.kernel()
    qc = uc.quasi_canonical
    from socutils.zmr import fock
    from socutils.zmr.active.sectors import ActiveSpace
    act = ActiveSpace.from_eris(qc.eris)
    ctx = classes.Context(qc.eris, qc.eps_core, qc.eps_virt,
                          fock.generalized_fock(qc.eris, 0 * uc.gamma))
    with pytest.raises(laplace_engine.IntruderError):
        laplace_engine.laplace_class(classes.CLASS_BY_NAME['cava'], ctx, act,
                                     uc.civecs, 2, numpy.array([100.0]), 20)

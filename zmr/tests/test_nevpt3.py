'''Multi-state NEVPT3 (zmr.pt4.nevpt3) against the third order of NEVPT4(SD)
and the determinant-basis reference.'''
import numpy
import pytest

from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.pt4 import NEVPT3, NEVPT4SD
from socutils.zmr.reference import nevpt4_bruteforce
from socutils.zmr.tests._helpers import model_eris

CASES = [((2, 4, 3), 2, 2), ((3, 4, 2), 3, 2), ((2, 3, 3), 1, 1), ((2, 4, 3), 2, 4)]
MODES = [('state', 'state'), ('state', 'union'), ('average', 'union'), ('state', 'none'),
         ('average', 'none')]


def _setup(sizes, nelec, nstate, seed=11):
    eris = model_eris(*sizes, seed=seed)
    return eris, ActiveSpace.from_eris(eris).solve(nelec, nstate)[1]


@pytest.mark.parametrize('sizes, nelec, nstate', CASES)
@pytest.mark.parametrize('e0, contraction', MODES)
def test_against_nevpt4(sizes, nelec, nstate, e0, contraction):
    eris, c = _setup(sizes, nelec, nstate)
    pt3 = NEVPT3(eris, nelec, civecs=c, e0=e0, contraction=contraction)
    pt3.verbose = 0
    pt3.kernel()
    pt4 = NEVPT4SD(eris, nelec, civecs=c, e0=e0, contraction=contraction)
    pt4.verbose = 0
    pt4.kernel()
    for a, b in ((pt3.h2, pt4.h2), (pt3.h3, pt4.h3), (pt3.e2, pt4.e2), (pt3.e3, pt4.e3)):
        assert abs(a - b).max() < 1e-11
    ref = nevpt4_bruteforce.kernel(eris, nelec, c, e0, contraction)
    assert abs(pt3.e3 - ref[1]).max() < 1e-9


def test_third_order_does_not_depend_on_contraction():
    eris, c = _setup((2, 4, 3), 2, 3)
    e = []
    for contraction in ('union', 'none'):
        pt = NEVPT3(eris, 2, civecs=c, contraction=contraction)
        pt.verbose = 0
        e.append(pt.kernel())
    assert abs(e[0] - e[1]).max() < 1e-10

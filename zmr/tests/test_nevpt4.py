'''Multi-state NEVPT4(SD) against the determinant-basis reference and, at
second order, against the existing NEVPT2 programs.'''
import numpy
import pytest

from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.pt4 import NEVPT4SD
from socutils.zmr.reference import nevpt4_bruteforce
from socutils.zmr.tests._helpers import model_eris
from socutils.zmr.uc.nevpt2 import UCNEVPT2

CASES = [((2, 4, 3), 2, 2), ((3, 4, 2), 3, 2), ((2, 3, 3), 1, 1)]
MODES = [('state', 'state', False), ('state', 'union', False), ('average', 'union', False),
         ('state', 'none', False), ('average', 'none', False), ('average', 'union', True),
         ('state', 'none', True)]


def _setup(sizes, nelec, nstate, seed=11):
    eris = model_eris(*sizes, seed=seed)
    return eris, ActiveSpace.from_eris(eris).solve(nelec, nstate)[1]


@pytest.mark.parametrize('sizes, nelec, nstate', CASES)
@pytest.mark.parametrize('e0, contraction, renorm', MODES)
def test_against_bruteforce(sizes, nelec, nstate, e0, contraction, renorm):
    eris, c = _setup(sizes, nelec, nstate)
    pt = NEVPT4SD(eris, nelec, civecs=c, e0=e0, contraction=contraction, renormalization=renorm)
    pt.verbose = 0
    pt.kernel()
    ref = nevpt4_bruteforce.kernel(eris, nelec, c, e0, contraction, renorm)
    for mine, theirs in zip((pt.e2, pt.e3, pt.e4), ref):
        assert abs(mine - theirs).max() < 1e-9


@pytest.mark.parametrize('sizes, nelec, nstate', CASES)
@pytest.mark.parametrize('e0', ['state', 'average'])
def test_second_order_is_nevpt2(sizes, nelec, nstate, e0):
    eris, c = _setup(sizes, nelec, nstate)
    for contraction, ref in (('union', PCNEVPT2(eris, nelec, civecs=c, e0=e0, contraction='union')),
                             ('state', PCNEVPT2(eris, nelec, civecs=c, e0=e0, contraction='state')),
                             ('none', UCNEVPT2(eris, nelec, civecs=c, e0=e0))):
        if contraction == 'state' and e0 == 'average':
            continue
        ref.verbose = 0
        e_ref = ref.kernel()
        pt = NEVPT4SD(eris, nelec, civecs=c, e0=e0, contraction=contraction)
        pt.verbose = 0
        pt.kernel()
        assert abs(pt.e2 - e_ref).max() < 1e-9, contraction


def test_one_state_series():
    '''One state: E3 = <x|V - E1|x> and E4 = <w|R|w> with the Rayleigh-Schroedinger
    expressions, and the contractions 'state' and 'union' coincide.'''
    eris, c = _setup((2, 4, 3), 2, 1)
    e = {}
    for contraction in ('state', 'union'):
        pt = NEVPT4SD(eris, 2, civecs=c, contraction=contraction)
        pt.verbose = 0
        pt.kernel()
        e[contraction] = (pt.e2, pt.e3, pt.e4)
    for a, b in zip(e['state'], e['union']):
        assert abs(a - b).max() < 1e-10


def test_average_union_is_rotation_invariant():
    eris, c = _setup((2, 4, 3), 2, 3)
    rng = numpy.random.default_rng(3)
    u, _ = numpy.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    out = []
    for cc in (c, c @ u):
        pt = NEVPT4SD(eris, 2, civecs=cc, e0='average', contraction='union')
        pt.verbose = 0
        pt.kernel()
        out.append(pt.e4)
    assert abs(out[0] - out[1]).max() < 1e-9

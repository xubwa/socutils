'''Active-space transition RDMs of rank 1-4 vs the reference determinant
module (operator strings by definition).'''
import itertools

import numpy
import pytest

from socutils.zmr.active import rdm
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.reference import dets
from socutils.zmr.tests._helpers import random_hamiltonian

NORB = 6


@pytest.fixture(scope='module')
def states():
    h, eri = random_hamiltonian(NORB, seed=41)
    act = ActiveSpace(h, eri)
    _, c = act.solve(3, 3)
    return act, c


def _ref(c, k, bra, ket, idx):
    strs = dets.gen_strings(range(NORB), 3)
    b = dets.SparseVector(strs, c[:, bra])
    t = dets.SparseVector(strs, c[:, ket])
    p = list(idx[0::2])
    q = list(idx[1::2])
    # p1^+ ... pk^+ qk ... q1
    orbs = p + q[::-1]
    daggers = (True,) * k + (False,) * k
    return dets.opstring_elements(daggers, [orbs], b, t)[0]


@pytest.mark.parametrize('k', [1, 2, 3, 4])
def test_rdm_by_definition(states, k):
    act, c = states
    dm = rdm.make_rdm(act, c, 3, k)
    rng = numpy.random.default_rng(k)
    for _ in range(40):
        idx = tuple(rng.integers(0, NORB, size=2 * k))
        for bra, ket in [(0, 0), (0, 2), (2, 1)]:
            assert abs(dm[(bra, ket) + idx] - _ref(c, k, bra, ket, idx)) < 1e-13


def test_rdm_conventions(states):
    act, c = states
    dm1 = rdm.make_rdm(act, c, 3, 1)
    dm2 = rdm.make_rdm(act, c, 3, 2)
    assert abs(dm1[0, 1] - act.trans_rdm1(c[:, 0], c[:, 1], 3)).max() < 1e-13
    # partial trace: sum_r <p^+ r^+ r q> = (N - 1) <p^+ q>
    assert abs(numpy.einsum('pqrr->pq', dm2[0, 0]) - 2 * dm1[0, 0]).max() < 1e-12
    # hermiticity of transition RDMs: dm^{IJ}[p,q,...] = conj(dm^{JI}[q,p,...])
    dm3 = rdm.make_rdm(act, c, 3, 3)
    assert abs(dm3[0, 2] - dm3[2, 0].transpose(1, 0, 3, 2, 5, 4).conj()).max() < 1e-13


def test_rdm_beyond_electron_number(states):
    act, c = states
    assert abs(rdm.make_rdm(act, c, 3, 4)).max() == 0

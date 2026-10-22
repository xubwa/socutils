import numpy
import pytest
from socutils.fci import zfci

from socutils.zmr.active import strings, sectors
from socutils.zmr.interface import pyscf_nr
from socutils.zmr.reference import dets
from socutils.zmr.tests import systems
from socutils.zmr.tests._helpers import random_hamiltonian, random_vector

NORB = 6


@pytest.fixture(scope='module')
def act():
    h, eri = random_hamiltonian(NORB, seed=21)
    return sectors.ActiveSpace(h, eri, e_core=0.25)


def test_strings_match_reference():
    for n in range(NORB + 1):
        assert numpy.array_equal(strings.gen_strings(NORB, n), dets.gen_strings(range(NORB), n))
    assert len(strings.gen_strings(NORB, -1)) == 0
    assert len(strings.gen_strings(NORB, NORB + 1)) == 0


@pytest.mark.parametrize('nelec', range(NORB + 1))
def test_sector_hamiltonian_matches_zfci(act, nelec):
    ref = zfci.make_hmat(act.heff, act.aaaa, NORB, nelec)
    assert abs(act.make_hmat(nelec) - ref).max() < 1e-12


@pytest.mark.parametrize('nelec', range(NORB + 1))
def test_single_operators_match_reference(act, nelec):
    src = dets.gen_strings(range(NORB), nelec)
    c = random_vector(len(src), seed=nelec)
    for dagger, apply in [(True, act.apply_cre), (False, act.apply_des)]:
        out = apply(c, nelec)
        dst = dets.gen_strings(range(NORB), nelec + (1 if dagger else -1))
        for p in range(NORB):
            ref = dets.operator_matrix([((dagger,), [[p]], [1.0])], dst, src) @ c
            assert abs(out[p] - ref).max(initial=0) < 1e-14


def test_composite_string_and_batches(act):
    '''x^+ y^+ z |c> built from single operators, for a batch of two vectors,
    against the reference operator strings.'''
    nelec = 3
    src = dets.gen_strings(range(NORB), nelec)
    dst = dets.gen_strings(range(NORB), nelec + 1)
    c = numpy.stack([random_vector(len(src), seed=1), random_vector(len(src), seed=2)])
    v = act.apply_des(c, nelec)            # (2, z, N-1)
    v = act.apply_cre(v, nelec - 1)        # (2, z, y, N)
    v = act.apply_cre(v, nelec)            # (2, z, y, x, N+1)
    rng = numpy.random.default_rng(0)
    for x, y, z in rng.integers(0, NORB, size=(30, 3)):
        m = dets.operator_matrix([((True, True, False), [[x, y, z]], [1.0])], dst, src)
        for b in range(2):
            assert abs(v[b, z, y, x] - m @ c[b]).max() < 1e-14


def test_rdm1(act):
    nelec = 3
    e, c = act.solve(nelec, 2)
    strs = dets.gen_strings(range(NORB), nelec)
    bra = dets.SparseVector(strs, c[:, 0])
    ket = dets.SparseVector(strs, c[:, 1])
    assert abs(act.trans_rdm1(c[:, 0], c[:, 1], nelec) - dets.trans_rdm1(bra, ket, NORB)).max() < 1e-13
    g = act.averaged_rdm1(c, nelec, weights=[0.3, 0.7])
    ref = 0.3 * dets.trans_rdm1(bra, bra, NORB) + 0.7 * dets.trans_rdm1(ket, ket, NORB)
    assert abs(g - ref).max() < 1e-13
    assert abs(numpy.trace(g) - nelec) < 1e-12


def test_sector_spectrum(act):
    for nelec in range(NORB + 1):
        sec = act.sector(nelec)
        h = act.make_hmat(nelec)
        assert abs(h @ sec.vecs - sec.vecs * sec.lam).max() < 1e-11


def test_casci_energy_h2o():
    mc = systems.h2o_casci(4, 4, nroots=3)
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=1)
    act = sectors.ActiveSpace.from_eris(eris)
    e, _ = act.solve(nelec, 1)
    assert abs(e[0] - mc.e_tot[0]) < 1e-10

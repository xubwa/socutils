'''zmr.dpd against dense numpy arrays: round trip, sort, contract, for a cyclic group with
parity (C2h*), a linear group (Coov*, unbounded n), the trivial group and a leg without
symmetry.'''
import numpy
import pytest

from socutils.zmr import dpd

# fermion irreps (ir = 2 n + p): C2h n = 1, 3 (mod 4), p = 0, 1; Coov n = 2 m_j
IRREPS = {'C2h': [2, 3, 6, 7], 'Coov': [-10, -6, -2, 2, 6, 10], None: [0]}


def random_leg(group, rng, name=''):
    irs = IRREPS[group]
    k = rng.integers(1, len(irs) + 1)
    chosen = sorted(rng.choice(irs, size=k, replace=False))
    return dpd.Leg(chosen, rng.integers(1, 4, size=k), name)


def orbital_irreps(leg):
    return numpy.repeat(numpy.asarray(leg.irreps), leg.sizes)


def allowed(legs, dirs, group, irrep=0):
    g = dpd.Group(group)
    rank = len(legs)
    arrays = []
    for k, leg in enumerate(legs):
        shape = [1] * rank
        shape[k] = leg.size
        arrays.append(numpy.broadcast_to(orbital_irreps(leg).reshape(shape),
                                         [x.size for x in legs]))
    return numpy.asarray(g.combine(arrays, dirs)) == irrep


def random_tensor(legs, dirs, split, group, rng):
    shape = [x.size for x in legs]
    a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    a = numpy.where(allowed(legs, dirs, group), a, 0)
    return a, dpd.Tensor.from_dense(a, legs, dirs, split, group)


@pytest.mark.parametrize('group', ['C2h', 'Coov', None])
def test_round_trip_and_sort(group):
    rng = numpy.random.default_rng(1)
    for _ in range(5):
        legs = [random_leg(group, rng) for _ in range(4)]
        dirs = (-1, -1, 1, 1)
        a, t = random_tensor(legs, dirs, 2, group, rng)
        assert abs(t.to_dense() - a).max() < 1e-14
        for order, split in (((2, 3, 0, 1), 2), ((0, 2, 1, 3), 1), ((3, 1, 0, 2), 3)):
            s = dpd.sort(t, order, split)
            assert abs(s.to_dense() - a.transpose(order)).max() < 1e-14


@pytest.mark.parametrize('group', ['C2h', 'Coov', None])
def test_contract(group):
    rng = numpy.random.default_rng(2)
    for _ in range(5):
        i, j, a, b, c = (random_leg(group, rng, n) for n in 'ijabc')
        # T[i, j, a, b] (holes annihilated, particles created) times V[a, b, c, j']
        ta, t = random_tensor([i, j, a, b], (-1, -1, 1, 1), 2, group, rng)
        va, v = random_tensor([a, b, c, j], (-1, -1, 1, 1), 2, group, rng)
        out = dpd.contract(t, v)
        ref = numpy.einsum('ijab,abcd->ijcd', ta, va)
        assert abs(out.to_dense() - ref).max() < 1e-12
        # one contracted leg (b), after sorts of both operands
        t1 = dpd.sort(t, (0, 1, 2, 3), 3)
        v1 = dpd.sort(v, (1, 0, 2, 3), 1)                 # legs b | a, c, j
        out = dpd.contract(t1, v1)
        ref = numpy.einsum('ijab,ebcd->ijaecd', ta, va)
        assert abs(out.to_dense() - ref).max() < 1e-12


def test_contract_one_leg():
    rng = numpy.random.default_rng(3)
    group = 'Coov'
    i, j, a, b, c = (random_leg(group, rng, n) for n in 'ijabc')
    xa, x = random_tensor([i, j, a, b], (-1, -1, 1, 1), 3, group, rng)
    # F[b, c] with b annihilated (opposite to the created b of x), c created
    fa, f = random_tensor([b, c], (-1, 1), 1, group, rng)
    out = dpd.contract(x, f)
    ref = numpy.einsum('ijab,bc->ijac', xa, fa)
    assert abs(out.to_dense() - ref).max() < 1e-12


def test_unsymmetric_leg():
    rng = numpy.random.default_rng(4)
    p = dpd.Leg.unsymmetric(5)
    q, r = random_leg('Coov', rng), random_leg('Coov', rng)
    a = rng.normal(size=(5, q.size, r.size)) + 0j
    t = dpd.Tensor.from_dense(a, [p, q, r], (1, -1, 1), 1, 'Coov')
    assert abs(t.to_dense() - a).max() == 0          # nothing dropped
    assert abs(dpd.sort(t, (1, 2, 0), 2).to_dense() - a.transpose(1, 2, 0)).max() == 0


def test_leg_from_orbsym():
    leg = dpd.Leg.from_orbsym([2, 2, -2, -2, -2, 6])
    assert leg.irreps == (2, -2, 6) and leg.sizes == (2, 3, 1)
    with pytest.raises(ValueError):
        dpd.Leg.from_orbsym([2, -2, 2])

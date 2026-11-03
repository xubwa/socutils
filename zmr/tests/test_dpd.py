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


@pytest.mark.parametrize('group', ['C2h', 'Coov'])
def test_einsum(group):
    rng = numpy.random.default_rng(5)
    i, j, a, b, c = (random_leg(group, rng, n) for n in 'ijabc')
    ta, t = random_tensor([i, j, a, b], (-1, -1, 1, 1), 2, group, rng)
    fa, f = random_tensor([b, c], (-1, 1), 1, group, rng)
    va, v = random_tensor([a, b, c, j], (-1, -1, 1, 1), 2, group, rng)
    out = dpd.einsum('ijab,bc->ijac', t, f)
    assert abs(out.to_dense() - numpy.einsum('ijab,bc->ijac', ta, fa)).max() < 1e-12
    # output letters in another order, accumulated
    acc = dpd.einsum('ijab,bc->cjia', t, f, split=2)
    dpd.einsum('ijab,bc->cjia', t, f, out=acc, alpha=-0.5)
    ref = 0.5 * numpy.einsum('ijab,bc->cjia', ta, fa)
    assert abs(acc.to_dense() - ref).max() < 1e-12
    out = dpd.einsum('ijab,abcd->ijcd', t, v)
    assert abs(out.to_dense() - numpy.einsum('ijab,abcd->ijcd', ta, va)).max() < 1e-12


@pytest.mark.parametrize('group', ['C2h', 'Coov'])
def test_contract_symmetric(group):
    '''sum_P conj(B[P, p, r]) B[P, q, s] with an auxiliary index P without symmetry
    (B = U X, X adapted, U orthogonal): only the allowed matrices are formed.'''
    rng = numpy.random.default_rng(6)
    boson = {'C2h': [0, 1, 4, 5], 'Coov': [-8, -4, 0, 4, 8]}[group]
    k = dpd.Leg(boson, rng.integers(1, 4, size=len(boson)))
    p, r = random_leg(group, rng), random_leg(group, rng)
    x = rng.normal(size=(k.size, p.size, r.size)) + 1j * rng.normal(size=(k.size, p.size, r.size))
    x = numpy.where(allowed([k, p, r], (1, -1, 1), group), x, 0)
    u = numpy.linalg.qr(rng.normal(size=(k.size, k.size)))[0]
    bmat = numpy.einsum('Pk,kpr->Ppr', u, x)
    P = dpd.Leg.unsymmetric(k.size)
    left = dpd.Tensor.from_dense(bmat.conj().transpose(1, 2, 0), [p, r, P], (1, -1, 1), 2, group)
    right = dpd.Tensor.from_dense(bmat, [P, p, r], (-1, -1, 1), 1, group)
    st = dpd.Tensor([p, r, p, r], (1, -1, -1, 1), 2, group).structure
    out = dpd.contract_symmetric(left, right, st)
    ref = numpy.einsum('Ppr,Pqs->prqs', bmat.conj(), bmat)
    assert abs(out.to_dense() - ref).max() < 1e-12


def test_identity_and_map_blocks():
    rng = numpy.random.default_rng(7)
    leg = random_leg('Coov', rng)
    e = dpd.identity(leg, 'Coov')
    assert abs(e.to_dense() - numpy.eye(leg.size)).max() == 0
    eps = rng.normal(size=leg.size)
    a, t = random_tensor([leg, leg], (-1, 1), 1, 'Coov', rng)

    def scale(sec, blk):
        blk *= (eps[leg.offsets[sec[0]]:leg.offsets[sec[0] + 1], None]
                - eps[None, leg.offsets[sec[1]]:leg.offsets[sec[1] + 1]])
    dpd.map_blocks(t, scale)
    assert abs(t.to_dense() - a * (eps[:, None] - eps[None, :])).max() < 1e-13

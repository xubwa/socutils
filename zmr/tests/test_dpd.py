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


@pytest.mark.parametrize('group', ['C2h', 'Coov'])
def test_boxes(group):
    '''Every block of a tensor through its box (base, strides) equals the dense block,
    and the strided box copy reads, writes and adds them, also through a permutation.'''
    from socutils.zmr.lib import kernels
    rng = numpy.random.default_rng(9)
    legs = [random_leg(group, rng) for _ in range(4)]
    a, t = random_tensor(legs, (-1, -1, 1, 1), 2, group, rng)
    boxes = t.boxes()
    assert boxes
    order = (2, 0, 3, 1)
    base, stride, dims, doff, ld, ref = [], [], [], [], [], []
    off = 0
    for sec, (b0, st, sz) in boxes.items():
        sl = tuple(slice(l.offsets[x], l.offsets[x + 1]) for l, x in zip(legs, sec))
        blk = a[sl].transpose(order)
        base.append(b0)
        stride.append([st[k] for k in order])
        dims.append([sz[k] for k in order])
        nr, nc = sz[order[0]] * sz[order[1]], sz[order[2]] * sz[order[3]]
        doff.append(off)
        ld.append(nc)
        ref.append(blk.reshape(nr, nc))
        off += nr * nc
    mat = numpy.empty(off, dtype=complex)
    kernels.box_copy_strided(0, t.data, mat, 2, base, stride, dims, doff, ld, conj=True)
    for r, d0, n in zip(ref, doff, ld):
        assert abs(mat[d0:d0 + r.size].reshape(-1, n) - r.conj()).max() == 0
    t2 = t.copy()
    kernels.box_copy_strided(2, t2.data, mat, 2, base, stride, dims, doff, ld, alpha=0.5,
                             conj=True)
    assert abs(t2.to_dense() - 1.5 * a).max() < 1e-14
    kernels.box_copy_strided(1, t2.data, mat, 2, base, stride, dims, doff, ld, conj=True)
    assert abs(t2.to_dense() - a).max() < 1e-14


@pytest.mark.parametrize('group', ['C2h', 'Coov'])
def test_view_and_add_permuted(group):
    rng = numpy.random.default_rng(10)
    p, q, r, s = (random_leg(group, rng) for _ in range(4))
    a, t = random_tensor([p, q, r, s], (-1, -1, 1, 1), 2, group, rng)
    v = dpd.View(t, (2, 3, 0, 1), conj=True)
    assert abs(numpy.asarray(v) - a.transpose(2, 3, 0, 1).conj()).max() == 0
    assert abs(numpy.asarray(v.transpose((1, 0, 2, 3))) -
               a.transpose(2, 3, 0, 1).conj().transpose(1, 0, 2, 3)).max() == 0
    # the Hermitian partner block: X[rs, pq] += conj(X[pq, rs]) (directions reversed by
    # the conjugation, restored by the permutation)
    b, u = random_tensor([r, s, p, q], (-1, -1, 1, 1), 2, group, rng)
    dpd.add_permuted(u, t, (2, 3, 0, 1), alpha=0.5, conj=True)
    assert abs(u.to_dense() - (b + 0.5 * a.transpose(2, 3, 0, 1).conj())).max() < 1e-14
    dpd.add_permuted(u, t, (2, 3, 0, 1), conj=True, assign=True)
    assert abs(u.to_dense() - a.transpose(2, 3, 0, 1).conj()).max() < 1e-14


@pytest.mark.parametrize('kind', ['xor', 'u1xu1', 'real'])
def test_other_groups(kind):
    '''The same tensors for a PySCF single group (XOR), a product of two additive groups
    (e.g. m_l and 2 S_z of spin orbitals), and real data: round trip, sort, contract.'''
    rng = numpy.random.default_rng(11)
    if kind in ('xor', 'real'):
        group, irs = dpd.XorGroup(), [0, 1, 2, 3, 5, 7]
    else:
        group = dpd.ProductGroup(dpd.U1Group(), dpd.U1Group())
        irs = [(m, s) for m in (-2, 0, 2) for s in (-1, 1)]
    def leg():
        k = rng.integers(1, len(irs) + 1)
        chosen = [irs[i] for i in sorted(rng.choice(len(irs), size=k, replace=False))]
        return dpd.Leg(chosen, rng.integers(1, 4, size=k))
    def mask(legs, dirs):
        shape = [x.size for x in legs]
        out = numpy.zeros(shape, dtype=bool)
        orb = [[ir for ir, n in zip(x.irreps, x.sizes) for _ in range(n)] for x in legs]
        for idx in numpy.ndindex(*shape):
            out[idx] = group.combine([orb[k][i] for k, i in enumerate(idx)], dirs) in (0, (0, 0))
        return out
    i, j, a, b, c = (leg() for _ in range(5))
    real = kind == 'real'
    def rnd(legs, dirs):
        shape = [x.size for x in legs]
        x = rng.normal(size=shape) + (0 if real else 1j * rng.normal(size=shape))
        x = numpy.where(mask(legs, dirs), x, 0)
        return x, dpd.Tensor.from_dense(x, legs, dirs, 2, group)
    ta, t = rnd([i, j, a, b], (-1, -1, 1, 1))
    va, v = rnd([a, b, c, j], (-1, -1, 1, 1))
    assert t.dtype == (numpy.float64 if real else numpy.complex128)
    assert abs(t.to_dense() - ta).max() < 1e-14
    assert abs(dpd.sort(t, (2, 3, 0, 1), 2).to_dense() - ta.transpose(2, 3, 0, 1)).max() < 1e-14
    out = dpd.einsum('ijab,abcd->ijcd', t, v)
    assert abs(out.to_dense() - numpy.einsum('ijab,abcd->ijcd', ta, va)).max() < 1e-12


@pytest.mark.parametrize('group', ['C2h', 'Coov'])
def test_dense_conversions(group):
    rng = numpy.random.default_rng(12)
    legs = [random_leg(group, rng) for _ in range(4)]
    a, t = random_tensor(legs, (-1, -1, 1, 1), 2, group, rng)
    u = dpd.Tensor(legs, (-1, -1, 1, 1), 2, group)
    dpd.dense_to_tensor(a, u)
    assert abs(u.data - t.data).max() == 0
    assert abs(dpd.tensor_to_dense(t) - a).max() == 0
    b = rng.normal(size=a.shape) + 0j
    b0 = b.copy()
    dpd.tensor_to_dense(t, b, alpha=2.0, beta=1.0)
    assert abs(b - (b0 + 2 * a)).max() < 1e-14


@pytest.mark.parametrize('group', ['C2h', 'Coov', None])
def test_permuted_conversions_and_flipped(group):
    rng = numpy.random.default_rng(13)
    legs = [random_leg(group, rng) for _ in range(4)]
    a, t = random_tensor(legs, (-1, -1, 1, 1), 2, group, rng)
    perm = (2, 0, 3, 1)
    # a stored with its axes permuted: axis perm[k] of x is leg k of t
    x = numpy.ascontiguousarray(a.transpose(numpy.argsort(perm)))
    u = dpd.Tensor(legs, (-1, -1, 1, 1), 2, group)
    dpd.dense_to_tensor(x, u, perm=perm)
    assert abs(u.data - t.data).max() == 0
    # read through a transposed view in place
    u = dpd.Tensor(legs, (-1, -1, 1, 1), 2, group)
    dpd.dense_to_tensor(x.transpose(perm), u)
    assert abs(u.data - t.data).max() == 0
    # an array in Fortran order (read in place)
    u = dpd.Tensor(legs, (-1, -1, 1, 1), 2, group)
    xf = numpy.asfortranarray(x)
    flat, _, _ = dpd._flat(xf.transpose(perm), u.data.dtype)
    assert numpy.shares_memory(flat, xf)
    dpd.dense_to_tensor(xf, u, perm=perm)
    assert abs(u.data - t.data).max() == 0
    # an owner of permuted axes (a ufunc of a transposed array keeps its layout)
    xp = 1.0 * numpy.ascontiguousarray(a).transpose(1, 3, 0, 2)
    assert xp.base is None and not xp.flags.c_contiguous and not xp.flags.f_contiguous
    flat, _, _ = dpd._flat(xp, u.data.dtype)
    assert numpy.shares_memory(flat, xp)
    u = dpd.Tensor(legs, (-1, -1, 1, 1), 2, group)
    dpd.dense_to_tensor(xp, u, perm=(2, 0, 3, 1))
    assert abs(u.data - t.data).max() == 0
    yf = numpy.zeros_like(xf)
    dpd.tensor_to_dense(t, yf, perm=perm)
    assert abs(yf - x).max() == 0
    y = numpy.zeros_like(x)
    dpd.tensor_to_dense(t, y, perm=perm, conj=True, beta=1.0)
    assert abs(y - x.conj()).max() == 0
    z = dpd.tensor_to_dense(t, perm=perm)
    assert abs(z - x).max() == 0
    # all directions reversed: the same elements in the same buffer
    f = t.flipped()
    assert f.data is t.data and f.dirs == (1, 1, -1, -1)
    assert abs(f.to_dense() - a).max() == 0


@pytest.mark.parametrize('dtype', [numpy.float64, numpy.complex128])
def test_gemm_batch(dtype):
    from socutils.zmr.lib import linalg as la
    rng = numpy.random.default_rng(14)
    rnd = lambda *s: (rng.normal(size=s) + (1j * rng.normal(size=s) if dtype == numpy.complex128
                                            else 0)).astype(dtype)
    shapes = [(3, 4, 5), (1, 7, 2), (6, 0, 3), (5, 5, 5), (0, 3, 2)]
    prods = [(rnd(m, k), rnd(k, n), rnd(m, n)) for m, k, n in shapes]
    want = [2.0 * c + a @ b for a, b, c in prods]
    la.gemm_batch(prods, beta=2.0)
    for (_, _, c), w in zip(prods, want):
        assert c.size == 0 or abs(c - w).max() < 1e-12

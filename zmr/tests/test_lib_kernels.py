'''OpenMP kernels of libzmr vs their numpy reference implementations.'''
import numpy
import pytest

from socutils.zmr import lib
from socutils.zmr.active import strings
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.lib import kernels
from socutils.zmr.tests._helpers import random_hamiltonian

pytestmark = pytest.mark.skipif(lib.zmr() is None, reason='libzmr is not built (run make)')


def _rand(shape, seed):
    rng = numpy.random.default_rng(seed)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _reference(func, *args):
    '''Run a kernel wrapper with the native library disabled.'''
    import os
    os.environ['ZMR_NO_NATIVE'] = '1'
    lib.reset()
    try:
        return func(*args)
    finally:
        del os.environ['ZMR_NO_NATIVE']
        lib.reset()


@pytest.mark.parametrize('dagger', [True, False])
def test_apply_op(dagger):
    norb, nelec = 7, 3
    target, sign = strings.single_op_table(norb, nelec, dagger)
    ndst = len(strings.gen_strings(norb, nelec + (1 if dagger else -1)))
    vecs = _rand((2, 3, target.shape[1]), 1)
    out = kernels.apply_op(target, sign, vecs, ndst)
    ref = _reference(kernels.apply_op, target, sign, vecs, ndst)
    assert out.shape == ref.shape == (2, 3, norb, ndst)
    assert abs(out - ref).max() == 0


def test_apply_strings():
    '''Strings of three operators vs chained single-operator tables, and
    writing a block of functions at an offset of the output.'''
    norb, nelec = 7, 3
    h, eri = random_hamiltonian(norb, seed=2)
    act = ActiveSpace(h, eri)
    c = _rand((3, act.dim(nelec)), 6)
    x3 = act.apply_cre(act.apply_cre(act.apply_des(c, nelec), nelec - 1), nelec)   # [v, u, t]
    ref = numpy.moveaxis(x3, (1, 2, 3), (3, 2, 1)).reshape(3, norb**3, -1)         # t^+ u^+ v
    ops = numpy.indices((norb,) * 3).reshape(3, -1).T
    out = kernels.apply_strings(norb, nelec, (True, True, False), ops, c)
    assert out.shape == ref.shape
    assert abs(out - ref).max() == 0
    assert abs(out - _reference(kernels.apply_strings, norb, nelec, (True, True, False),
                                ops, c)).max() == 0
    # a second block behind the first one
    buf = numpy.full((3, 5 + norb, ref.shape[2]), numpy.nan, dtype=complex)
    kernels.apply_strings(norb, nelec, (True, True, False), ops[40:45], c, buf, 0)
    kernels.apply_strings(norb, nelec, (True,), numpy.arange(norb)[:, None], c, buf, 5)
    assert abs(buf[:, :5] - ref[:, 40:45]).max() == 0
    assert abs(buf[:, 5:] - act.apply_cre(c, nelec)).max() == 0


def test_apply_strings_identity_and_empty_sector():
    norb, nelec = 5, 2
    c = _rand((2, 10), 7)
    out = kernels.apply_strings(norb, nelec, (), numpy.zeros((1, 0)), c)
    assert abs(out[:, 0] - c).max() == 0
    # three annihilators on two electrons
    out = kernels.apply_strings(norb, nelec, (False,) * 3, numpy.zeros((4, 3)), c)
    assert out.shape == (2, 4, 0)


@pytest.mark.parametrize('distinct', [False, True])
def test_weighted_gram(distinct):
    ns, nk, ne = 4, 9, 13
    u = _rand((ns, nk, ne), 2)
    rng = numpy.random.default_rng(3)
    lam = rng.uniform(0.5, 2.0, nk)
    delta = rng.uniform(1.0, 5.0, ne)
    shifts = rng.uniform(-0.3, 0.3, ns) if distinct else numpy.full(ns, 0.1)
    a = kernels.weighted_gram(u, lam, delta, shifts)
    ref = _reference(kernels.weighted_gram, u, lam, delta, shifts)
    assert abs(a - ref).max() < 1e-13
    # by definition
    d = numpy.zeros((ns, ns), dtype=complex)
    for j in range(ns):
        d[:, j] = -numpy.einsum('ike,ke->i', u.conj(), u[j] / (lam[:, None] + delta[None, :] - shifts[j]))
    assert abs(a - d).max() < 1e-13


@pytest.mark.parametrize('distinct', [False, True])
def test_contract_t(distinct):
    ns, nk, nx, ne = 4, 9, 5, 13
    t = _rand((ns, nk, nx), 12)
    t[1, 2:5] = 0                  # vanishing rows (another symmetry block)
    t[:, 7] = 0
    t[2, 0, 1:] = 0
    v = _rand((ne, nx), 13)
    rng = numpy.random.default_rng(14)
    lam = rng.uniform(0.5, 2.0, nk)
    delta = rng.uniform(1.0, 5.0, ne)
    shifts = numpy.array([0.1, 0.1, -0.2, 0.1]) if distinct else numpy.full(ns, 0.1)
    a = kernels.contract_t(t, v, lam, delta, shifts)
    u = numpy.einsum('jkx,ex->jke', t, v)
    tol = 1e-13 * abs(a).max()
    assert abs(a - kernels.weighted_gram(u, lam, delta, shifts)).max() < tol
    assert abs(a - _reference(kernels.contract_t, t, v, lam, delta, shifts)).max() < tol
    assert abs(kernels.contract_t(t[:, :0], v, lam[:0], delta, shifts)).max() == 0
    # one column only (state-specific first-order spaces)
    for f, x in ((kernels.contract_t, (t, v)), (kernels.weighted_gram, (u,))):
        one = f(*x, lam, delta, shifts, col=2)
        assert abs(one[:, 2] - a[:, 2]).max() < tol
        assert abs(numpy.delete(one, 2, axis=1)).max() == 0
        assert abs(one - _reference(f, *x, lam, delta, shifts, 2)).max() < tol


@pytest.mark.parametrize('real', [False, True])
@pytest.mark.parametrize('distinct', [False, True])
@pytest.mark.parametrize('col', [None, 1])
def test_gram_kernels_agree(real, distinct, col):
    '''The three forms of the denominator sum, for complex and real input,
    all columns or one.'''
    ns, nk, nx, ne = 4, 9, 5, 13
    t = _rand((ns, nk, nx), 21)
    v = _rand((ne, nx), 22)
    if real:
        t, v = t.real.copy(), v.real.copy()
    t[2, 3:6] = 0
    rng = numpy.random.default_rng(23)
    lam = rng.uniform(0.5, 2.0, nk)
    delta = rng.uniform(1.0, 5.0, ne)
    shifts = numpy.array([0.1, -0.2, -0.2, 0.3]) if distinct else numpy.full(ns, 0.1)
    u = numpy.einsum('jkx,ex->jke', t, v)
    ref = numpy.zeros((ns, ns), dtype=u.dtype)
    for j in range(ns) if col is None else [col]:
        ref[:, j] = -numpy.einsum('ike,ke->i', u.conj(), u[j] / (lam[:, None] + delta[None, :] - shifts[j]))
    tol = 1e-13 * abs(ref).max()
    for f, x in ((kernels.weighted_gram, (u,)), (kernels.contract_t, (t, v)),
                 (kernels.contract_w, (t, v))):
        a = f(*x, lam, delta, shifts, col=col)
        assert a.dtype == (numpy.float64 if real else numpy.complex128)
        assert abs(a - ref).max() < tol, f.__name__
        b = _reference(f, *x, lam, delta, shifts, col)
        assert b.dtype == a.dtype and abs(b - ref).max() < tol, f.__name__


@pytest.mark.parametrize('nk', [3, 400])
def test_gram_kernels_share_k_or_states(nk):
    '''Both ways of sharing the work among threads: over a long loop of
    first-order functions k, or over the states when there are few k.'''
    ns, nx, ne = 7, 4, 11
    t = _rand((ns, nk, nx), 15)
    t[3] = 0
    v = _rand((ne, nx), 16)
    rng = numpy.random.default_rng(17)
    lam = rng.uniform(0.5, 2.0, nk)
    delta = rng.uniform(1.0, 5.0, ne)
    shifts = rng.uniform(-0.3, 0.3, ns)
    u = numpy.einsum('jkx,ex->jke', t, v)
    ref = _reference(kernels.weighted_gram, u, lam, delta, shifts)
    tol = 1e-13 * abs(ref).max()
    assert abs(kernels.weighted_gram(u, lam, delta, shifts) - ref).max() < tol
    assert abs(kernels.contract_t(t, v, lam, delta, shifts) - ref).max() < tol


@pytest.mark.parametrize('nelec', range(0, 7))
def test_build_hmat(nelec):
    norb = 6
    h, eri = random_hamiltonian(norb, seed=4)
    act = ActiveSpace(h, eri)
    dim = act.dim(nelec)
    native = kernels.build_hmat(norb, nelec, dim, h, eri)
    assert abs(native - act.make_hmat_numpy(nelec)).max() < 1e-12


def test_fold_density():
    nao, n1, n2 = 6, 3, 4
    c = [_rand((nao, n), 5 + k) for k, n in enumerate((n1, n1, n2, n2))]
    out = kernels.fold_density(*c)
    ref = _reference(kernels.fold_density, *c)
    assert out.shape == (n1 * n2, nao * (nao + 1) // 2)
    assert abs(out - ref).max() < 1e-13


# ----------------------------------------------------------------------
# matrix-free H_act of the spinor code (kernels.sigma) against the dense
# sector Hamiltonian

def _sigma_integrals(norb, kind, seed=11):
    '''hermitian: only h = h^+, (pq|rs) = (rs|pq), (pq|rs)^* = (qp|sr);
    none: complex h and (pq|rs) without any symmetry.'''
    if kind == 'hermitian':
        return random_hamiltonian(norb, seed=seed)
    rng = numpy.random.default_rng(seed)
    h = rng.normal(size=(norb, norb)) + 1j * rng.normal(size=(norb, norb))
    eri = rng.normal(size=(norb,) * 4) + 1j * rng.normal(size=(norb,) * 4)
    return h, eri


@pytest.mark.parametrize('kind', ['hermitian', 'none'])
@pytest.mark.parametrize('norb, nelec', [(6, 0), (6, 1), (6, 2), (6, 3), (6, 5), (6, 6), (7, 4),
                                         (8, 3), (9, 7), (10, 5), (2, 2), (1, 1)])
def test_sigma(norb, nelec, kind):
    import math
    h, eri = _sigma_integrals(norb, kind)
    dim = math.comb(norb, nelec)
    hmat = kernels.build_hmat(norb, nelec, dim, h, eri)
    if kind == 'hermitian':
        assert abs(hmat - hmat.conj().T).max() < 1e-13
    tol = 1e-11 * max(abs(hmat).max(), 1.0)
    rng = numpy.random.default_rng(12)
    for nvec in (1, 2, 5, 37, 150):
        q = rng.normal(size=(dim, nvec)) + 1j * rng.normal(size=(dim, nvec))
        ref = hmat @ q
        out = kernels.sigma(norb, nelec, dim, h, eri, q)
        assert out.shape == q.shape and abs(out - ref).max() < tol
        if kind == 'hermitian':
            assert abs(kernels.sigma_det(norb, nelec, dim, h, eri, q) - ref).max() < tol
    # non-contiguous input: every second column of a Fortran-ordered array
    big = numpy.asfortranarray(rng.normal(size=(dim, 12)) + 1j * rng.normal(size=(dim, 12)))
    q = big[:, 1::2]
    assert not q.flags.c_contiguous
    assert abs(kernels.sigma(norb, nelec, dim, h, eri, q) - hmat @ q).max() < tol
    # real input, no vectors
    assert abs(kernels.sigma(norb, nelec, dim, h, eri, big.real) - hmat @ big.real).max() < tol
    assert kernels.sigma(norb, nelec, dim, h, eri, numpy.zeros((dim, 0))).shape == (dim, 0)


@pytest.mark.parametrize('threads', [1, 3, 16])
def test_sigma_threading(threads):
    '''Both threading modes (threads over batches of columns, threads over
    the strings with two electrons fewer).'''
    norb, nelec, dim = 12, 6, 924
    h, eri = random_hamiltonian(norb, seed=13)
    hmat = kernels.build_hmat(norb, nelec, dim, h, eri)
    rng = numpy.random.default_rng(14)
    before = lib.num_threads()
    lib.set_num_threads(threads)
    try:
        for nvec in (1, 2, 7, 15, 16, 17, 40, 131):
            q = rng.normal(size=(dim, nvec)) + 1j * rng.normal(size=(dim, nvec))
            ref = hmat @ q
            out = kernels.sigma(norb, nelec, dim, h, eri, q)
            assert abs(out - ref).max() < 1e-11 * abs(ref).max()
    finally:
        lib.set_num_threads(before)


def test_sigma_with_symmetry_charges():
    '''Integrals projected onto charge conservation (exact zeros): the
    matrix-free product equals the dense, block diagonal Hamiltonian.'''
    norb, nelec = 8, 4
    h, eri = random_hamiltonian(norb, seed=15)
    charges = numpy.array([0.5, -0.5, 1.5, -1.5, 0.5, -0.5, 0.5, -0.5])
    dense = ActiveSpace(h, eri, charges=charges)
    free = ActiveSpace(h, eri, charges=charges)
    free.dense_limit = 0
    assert dense.blocks(nelec)[1] > 1
    rng = numpy.random.default_rng(16)
    for nvec in (1, 20):
        q = rng.normal(size=(dense.dim(nelec), nvec)) + 1j * rng.normal(size=(dense.dim(nelec), nvec))
        ref = dense.hmat(nelec) @ q
        assert abs(free.apply_h(nelec, q) - ref).max() < 1e-11 * abs(ref).max()
    assert not free._hmat


@pytest.mark.parametrize('shape, perm', [((3, 4, 5), (2, 0, 1)), ((2, 3, 4, 5, 6), (4, 1, 0, 3, 2)),
                                         ((7,), (0,)), ((40, 30, 6), (1, 2, 0)), ((1, 5, 1), (1, 0, 2))])
def test_transpose_add(shape, perm):
    from socutils.zmr.lib import kernels
    rng = numpy.random.default_rng(0)
    src = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    ref = src.transpose(perm)
    dst = rng.normal(size=ref.shape) + 0j
    expect = dst + (0.5 - 2j) * ref
    assert abs(kernels.transpose_add(dst, src, perm, alpha=0.5 - 2j) - expect).max() < 1e-13
    assert abs(kernels.transposed(src, perm) - ref).max() == 0
    big = numpy.tile(src[None], (60,) + (1,) * len(shape))        # above the size threshold
    p2 = [0] + [p + 1 for p in perm]
    assert abs(kernels.transposed(big, p2) - big.transpose(p2)).max() == 0
    out = numpy.zeros(big.transpose(p2).shape, dtype=complex)
    kernels.transpose_add(out, big, p2)
    kernels.transpose_add(out, big, p2, alpha=-1j)
    assert abs(out - (1 - 1j) * big.transpose(p2)).max() < 1e-13


@pytest.mark.parametrize('lead, n, trail', [((), 5, ()), ((3,), 6, (4,)), ((2, 3), 4, ()), ((), 7, (2, 3)),
                                            ((4,), 2, (3,)), ((2,), 1, (2,))])
def test_pair_gather_and_scatter(lead, n, trail):
    from socutils.zmr.lib import kernels
    rng = numpy.random.default_rng(1)
    shape = lead + (n, n) + trail
    a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    i, j = numpy.triu_indices(n, k=1)
    first = len(lead)
    packed = kernels.pair_gather(a, first)
    ref = numpy.moveaxis(numpy.moveaxis(a, (first, first + 1), (0, 1))[i, j], 0, first)
    assert packed.shape == ref.shape and abs(packed - ref).max(initial=0) == 0
    full = kernels.pair_scatter(packed, first, n)
    # the scattered array is the antisymmetric one built from the upper triangle of a
    upper = numpy.zeros_like(numpy.moveaxis(a, (first, first + 1), (0, 1)))
    upper[i, j] = numpy.moveaxis(a, (first, first + 1), (0, 1))[i, j]
    upper = upper - upper.transpose((1, 0) + tuple(range(2, upper.ndim)))
    assert abs(numpy.moveaxis(full, (first, first + 1), (0, 1)) - upper).max(initial=0) == 0

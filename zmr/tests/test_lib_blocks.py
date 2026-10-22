'''Block structure from exact zeros: components, block-wise eigh and the
grouping of vectors by the blocks they live in.'''
import numpy
import pytest

from socutils.zmr import lib
from socutils.zmr.lib import blocks, kernels


def _rand(shape, seed):
    rng = numpy.random.default_rng(seed)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _block_matrix(sizes, seed):
    '''Hermitian matrix that is block diagonal after a random permutation of
    the indices; returns (matrix, label of every index).'''
    n = sum(sizes)
    a = numpy.zeros((n, n), dtype=complex)
    lab = numpy.zeros(n, dtype=int)
    off = 0
    for k, m in enumerate(sizes):
        b = _rand((m, m), seed + k)
        a[off:off + m, off:off + m] = b + b.conj().T
        lab[off:off + m] = k
        off += m
    perm = numpy.random.default_rng(seed).permutation(n)
    return a[numpy.ix_(perm, perm)], lab[perm]


def _same_partition(x, y):
    pairs = set(zip(x.tolist(), y.tolist()))
    return len(pairs) == len(set(x.tolist())) == len(set(y.tolist()))


@pytest.fixture(params=[False, True], ids=['native', 'fallback'])
def native(request, monkeypatch):
    if request.param:
        monkeypatch.setenv('ZMR_NO_NATIVE', '1')
    lib.reset()
    yield
    monkeypatch.delenv('ZMR_NO_NATIVE', raising=False)
    lib.reset()


def test_components(native):
    a, lab = _block_matrix([5, 1, 7, 3], 1)
    labels, ncomp = blocks.components(a)
    assert ncomp == 4
    assert _same_partition(labels, lab)
    assert labels[0] == 0                           # numbered by first appearance
    # either triangle is enough
    for tri in (numpy.tril(a), numpy.triu(a)):
        l2, n2 = blocks.components(tri)
        assert n2 == 4 and (l2 == labels).all()
    assert blocks.components(_rand((6, 6), 2))[1] == 1
    assert blocks.components(numpy.zeros((3, 3), dtype=complex))[1] == 3
    assert blocks.components(numpy.zeros((0, 0), dtype=complex))[1] == 0


def test_eigh_by_blocks(native):
    a, lab = _block_matrix([6, 2, 9, 1, 4], 3)
    w, v = blocks.eigh(numpy.tril(a))                 # lower triangle only
    assert abs(w - numpy.linalg.eigvalsh(a)).max() < 1e-12
    assert (numpy.diff(w) >= 0).all()
    assert abs(a @ v - v * w).max() < 1e-12
    assert abs(v.conj().T @ v - numpy.eye(len(w))).max() < 1e-12
    # every eigenvector lives on one block
    for k in range(v.shape[1]):
        assert len(set(lab[abs(v[:, k]) > 0].tolist())) == 1
    # a dense matrix takes the ordinary route
    d = _rand((8, 8), 4)
    d = d + d.conj().T
    assert abs(blocks.eigh(d)[0] - numpy.linalg.eigvalsh(d)).max() < 1e-12


def test_block_support(native):
    labels = numpy.array([0, 1, 1, 2, 0, 2, 2])
    rows = numpy.zeros((4, 7), dtype=complex)
    rows[0, [0, 4]] = 1.0, 2j
    rows[1, [1, 5]] = 1.0, -1.0
    rows[3, 6] = 1e-300
    touch = kernels.block_support(rows, labels, 3)
    assert touch.dtype == bool
    assert (touch == numpy.array([[1, 0, 0], [0, 1, 1], [0, 0, 0], [0, 0, 1]], dtype=bool)).all()


def test_function_groups(native):
    labels = numpy.array([0, 0, 1, 1, 2, 2, 3, 3, 4])
    rows = numpy.zeros((6, 9), dtype=complex)
    rows[0, 0] = rows[0, 1] = 1.0          # block 0
    rows[1, 2] = rows[1, 5] = 1.0          # links blocks 1 and 2
    rows[2, 4] = 2.0                       # block 2
    rows[4, 7] = 1.0                       # block 3; row 3 vanishes; block 4 untouched
    rows[5, 1] = 1j                        # block 0
    groups = blocks.function_groups(rows, labels, 5)
    got = sorted((tuple(i.tolist()), tuple(r.tolist())) for i, r in groups)
    assert got == [((0, 1), (0, 5)), ((2, 3, 4, 5), (1, 2)), ((6, 7), (4,))]
    # nothing to split: one row over everything, or a single component
    full = numpy.ones((1, 9), dtype=complex)
    assert blocks.function_groups(full, labels, 5) is None
    assert blocks.function_groups(rows, numpy.zeros(9, dtype=int), 1) is None

'''MKL wrappers vs numpy.'''
import numpy
import pytest

from socutils.zmr import lib
from socutils.zmr.lib import linalg


def _rand(shape, seed):
    rng = numpy.random.default_rng(seed)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


@pytest.mark.skipif(lib._disabled(), reason='native libraries disabled by ZMR_NO_NATIVE')
def test_mkl_is_loaded():
    '''The test environment provides libmkl_rt (the one PySCF links to).'''
    assert lib.mkl() is not None
    assert lib.num_threads() >= 1


@pytest.mark.parametrize('opa', 'NTC')
@pytest.mark.parametrize('opb', 'NTC')
def test_gemm(opa, opb):
    a = _rand((7, 5) if opa == 'N' else (5, 7), 1)
    b = _rand((5, 9) if opb == 'N' else (9, 5), 2)
    f = {'N': lambda x: x, 'T': lambda x: x.T, 'C': lambda x: x.conj().T}
    assert abs(linalg.gemm(a, b, opa, opb) - f[opa](a) @ f[opb](b)).max() < 1e-13


def test_gemm_accepts_noncontiguous_and_empty():
    a = _rand((6, 8), 3)[:, ::2]
    b = _rand((4, 5), 4)
    assert abs(linalg.gemm(a, b) - a @ b).max() < 1e-13
    assert linalg.gemm(numpy.zeros((0, 3)), numpy.zeros((3, 4))).shape == (0, 4)


def test_herk():
    a = _rand((6, 11), 5)
    assert abs(linalg.herk(a, 'N') - a @ a.conj().T).max() < 1e-13
    assert abs(linalg.herk(a, 'C') - a.conj().T @ a).max() < 1e-13
    # lower triangle only
    assert abs(linalg.herk(a, 'N', fill=False) - numpy.tril(a @ a.conj().T)).max() < 1e-13
    assert abs(linalg.herk(a, 'C', fill=False) - numpy.tril(a.conj().T @ a)).max() < 1e-13


def test_cholesky():
    a = _rand((8, 12), 10)
    g = a @ a.conj().T
    low = linalg.cholesky(numpy.tril(g))             # only the lower triangle is read
    assert abs(numpy.triu(low, 1)).max() == 0
    assert abs(low @ low.conj().T - g).max() < 1e-12
    assert linalg.is_positive_definite(numpy.tril(g))
    g[3, 3] -= 2 * numpy.linalg.eigvalsh(g)[-1]
    assert linalg.cholesky(g) is None
    assert not linalg.is_positive_definite(g)


def test_orthonormalize():
    q = numpy.linalg.qr(_rand((30, 7), 11))[0]
    x = q @ (numpy.eye(7) + 1e-4 * _rand((7, 7), 12))       # nearly orthonormal columns
    y = linalg.orthonormalize(x)
    assert abs(y.conj().T @ y - numpy.eye(7)).max() < 1e-14
    assert abs(y - q @ (q.conj().T @ y)).max() < 1e-14       # same span
    assert linalg.orthonormalize(numpy.zeros((5, 0))).shape == (5, 0)
    with pytest.raises(numpy.linalg.LinAlgError):
        linalg.orthonormalize(numpy.ones((5, 2)))


def test_eigh_and_lowest():
    a = _rand((40, 40), 6)
    a = a + a.conj().T
    w_ref = numpy.linalg.eigvalsh(a)
    w, v = linalg.eigh(a)
    assert abs(w - w_ref).max() < 1e-12
    assert abs(a @ v - v * w).max() < 1e-11
    assert abs(v.conj().T @ v - numpy.eye(40)).max() < 1e-12
    wl, vl = linalg.eigh_lowest(a, 5)
    assert abs(wl - w_ref[:5]).max() < 1e-12
    assert abs(a @ vl - vl * wl).max() < 1e-11
    # only the lower triangle is read
    b = numpy.tril(a) + numpy.triu(_rand((40, 40), 7), 1)
    assert abs(linalg.eigh(b)[0] - w_ref).max() < 1e-12


def test_fallback_matches(monkeypatch):
    a = _rand((12, 12), 9)
    a = a + a.conj().T
    ref = linalg.eigh(a)[0]
    monkeypatch.setenv('ZMR_NO_NATIVE', '1')
    lib.reset()
    try:
        assert lib.mkl() is None
        assert abs(linalg.eigh(a)[0] - ref).max() < 1e-12
        assert abs(linalg.gemm(a, a, 'C', 'N') - a.conj().T @ a).max() < 1e-12
        g = a @ a.conj().T
        assert abs(linalg.herk(a, 'N', fill=False) - numpy.tril(g)).max() < 1e-12
        low = linalg.cholesky(numpy.tril(g))
        assert abs(low @ low.conj().T - g).max() < 1e-10
        q = numpy.linalg.qr(a[:, :5])[0] @ (numpy.eye(5) + 1e-3 * a[:5, :5])
        y = linalg.orthonormalize(q)
        assert abs(y.conj().T @ y - numpy.eye(5)).max() < 1e-13
    finally:
        monkeypatch.delenv('ZMR_NO_NATIVE')
        lib.reset()


def _real(shape, seed):
    return numpy.random.default_rng(seed).normal(size=shape)


@pytest.fixture(params=[False, True], ids=['mkl', 'fallback'])
def backend(request, monkeypatch):
    if request.param:
        monkeypatch.setenv('ZMR_NO_NATIVE', '1')
    lib.reset()
    yield
    monkeypatch.delenv('ZMR_NO_NATIVE', raising=False)
    lib.reset()


def test_real_input_stays_real(backend):
    '''float64 arrays take the d-routines and return float64.'''
    for opa in 'NTC':
        for opb in 'NTC':
            x = _real((7, 5) if opa == 'N' else (5, 7), 31)
            y = _real((5, 9) if opb == 'N' else (9, 5), 32)
            c = linalg.gemm(x, y, opa, opb)
            assert c.dtype == numpy.float64
            assert abs(c - (x if opa == 'N' else x.T) @ (y if opb == 'N' else y.T)).max() < 1e-12
    a, b = _real((7, 5), 31), _real((5, 9), 32)
    # mixed real / complex input is complex
    assert linalg.gemm(a, b + 0j).dtype == numpy.complex128
    g = linalg.herk(a, 'N')
    assert g.dtype == numpy.float64 and abs(g - a @ a.T).max() < 1e-12
    assert abs(linalg.herk(a, 'C', fill=False) - numpy.tril(a.T @ a)).max() < 1e-12
    s = _real((30, 30), 33)
    s = s + s.T
    w, v = linalg.eigh(numpy.tril(s))
    assert v.dtype == numpy.float64
    assert abs(w - numpy.linalg.eigvalsh(s)).max() < 1e-12
    assert abs(s @ v - v * w).max() < 1e-11
    wl, vl = linalg.eigh_lowest(s, 4)
    assert vl.dtype == numpy.float64 and abs(wl - w[:4]).max() < 1e-12
    assert abs(s @ vl - vl * wl).max() < 1e-11
    p = a @ a.T + numpy.eye(7)
    low = linalg.cholesky(numpy.tril(p))
    assert low.dtype == numpy.float64 and abs(low @ low.T - p).max() < 1e-12
    assert linalg.is_positive_definite(p) and not linalg.is_positive_definite(p - 100 * numpy.eye(7))
    q = numpy.linalg.qr(_real((20, 6), 34))[0] @ (numpy.eye(6) + 1e-4 * _real((6, 6), 35))
    y = linalg.orthonormalize(q)
    assert y.dtype == numpy.float64 and abs(y.T @ y - numpy.eye(6)).max() < 1e-13


def test_numpy_blas_is_limited_inside_the_drivers():
    '''numpy's OpenBLAS runs on one thread inside the block and gets its
    threads back afterwards; MKL is not touched.'''
    threadpoolctl = pytest.importorskip('threadpoolctl')
    if lib.mkl() is None:
        pytest.skip('no MKL: numpy BLAS is the only one')

    def openblas_threads():
        return [m['num_threads'] for m in threadpoolctl.threadpool_info()
                if m['internal_api'] == 'openblas']

    before = openblas_threads()
    mkl_before = lib.num_threads()
    with lib.numpy_blas_single_thread():
        assert all(n == 1 for n in openblas_threads())
        assert lib.num_threads() == mkl_before
    assert openblas_threads() == before


def test_gemm_into_existing_array():
    from socutils.zmr.lib import linalg as la
    rng = numpy.random.default_rng(3)
    for dtype in (float, complex):
        a = rng.normal(size=(5, 7)).astype(dtype)
        b = rng.normal(size=(7, 4)).astype(dtype)
        if dtype is complex:
            a = a + 1j * rng.normal(size=a.shape)
            b = b + 1j * rng.normal(size=b.shape)
        out = numpy.ones((5, 4), dtype=dtype)
        assert la.gemm(a, b, out=out) is out
        assert abs(out - a @ b).max() < 1e-13
        la.gemm(a, b, out=out, beta=1.0)
        assert abs(out - 2 * (a @ b)).max() < 1e-13
        with pytest.raises(ValueError):
            la.gemm(a, b, out=numpy.ones((4, 5), dtype=dtype))


def test_transposed_takes_strided_views():
    from socutils.zmr.lib import kernels
    rng = numpy.random.default_rng(4)
    a = rng.normal(size=(20, 30, 8, 9)) + 1j * rng.normal(size=(20, 30, 8, 9))
    view = a.transpose(2, 3, 0, 1)[:, 1:, ::2]
    assert not view.flags.c_contiguous
    assert abs(kernels.transposed(view, (3, 0, 2, 1)) - view.transpose(3, 0, 2, 1)).max() == 0

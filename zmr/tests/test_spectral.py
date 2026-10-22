'''Orthonormal span and contracted space of a set of CI vectors.'''
import numpy
import pytest

from socutils.zmr.pc import spectral


def _rand(shape, seed):
    rng = numpy.random.default_rng(seed)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _unitary(n, seed):
    return numpy.linalg.qr(_rand((n, n), seed))[0]


def _functions(m, d, sing, seed):
    '''m functions in a d-dimensional space (as rows) with the given singular
    values; also returns the right singular vectors as columns (d, len(sing)).'''
    k = len(sing)
    u = _unitary(m, seed)[:, :k]
    v = _unitary(d, seed + 1)[:, :k]
    return (u * sing) @ v.T, v


def _projector(q):
    return q @ q.conj().T


@pytest.mark.parametrize('m, d', [(12, 40), (60, 40)])
def test_span_with_small_metric_eigenvalues(m, d):
    '''Metric eigenvalues from 1 down to 1e-14: the functions above lindep
    are kept and the basis is orthonormal to machine precision, although the
    metric has a condition number of 1e9 inside the kept space.'''
    sing = 10.0 ** -numpy.linspace(0, 7, 10)
    rows, v = _functions(m, d, sing, 1)
    q, full = spectral.orthonormal_span(rows, lindep=1e-10)
    kept = sing**2 > 1e-10
    assert not full
    assert q.shape == (d, kept.sum())
    assert abs(q.conj().T @ q - numpy.eye(q.shape[1])).max() < 1e-12
    assert abs(_projector(q) - _projector(v[:, kept])).max() < 1e-6


def test_span_of_fewer_functions_is_their_column_space():
    rows = _rand((5, 30), 3)
    q, full = spectral.orthonormal_span(rows)
    assert not full and q.shape == (30, 5)
    # every function f_X = rows[X] lies in the span
    assert abs(rows.T - q @ (q.conj().T @ rows.T)).max() < 1e-12


def test_full_and_empty_span():
    q, full = spectral.orthonormal_span(_rand((50, 20), 4))
    assert full and q is None
    q, full = spectral.orthonormal_span(numpy.zeros((3, 20), dtype=complex))
    assert not full and q.shape == (20, 0)
    q, full = spectral.orthonormal_span(numpy.zeros((30, 20), dtype=complex))
    assert not full and q.shape == (20, 0)


@pytest.mark.parametrize('m', [6, 45])
def test_contracted_space_diagonalizes_h_in_the_span(m):
    d, rank = 30, 6
    rows, v = _functions(m, d, numpy.linspace(1.0, 0.3, rank), 5)
    h = _rand((d, d), 6)
    h = h + h.conj().T
    calls = []

    def sector():
        calls.append(1)

    b, mu = spectral.contracted_space_det(rows, lambda: h, sector)
    assert not calls and b.shape == (d, rank)
    assert abs(b.conj().T @ b - numpy.eye(rank)).max() < 1e-12
    assert abs(b.conj().T @ h @ b - numpy.diag(mu)).max() < 1e-12
    assert abs(_projector(b) - _projector(v)).max() < 1e-12
    assert abs(mu - numpy.linalg.eigvalsh(v.conj().T @ h @ v)).max() < 1e-12


def test_contracted_space_of_a_complete_set_is_the_sector():
    d = 12
    h = _rand((d, d), 7)
    h = h + h.conj().T
    lam, vecs = numpy.linalg.eigh(h)

    class Sector:
        pass

    sec = Sector()
    sec.lam, sec.vecs = lam, vecs
    b, mu = spectral.contracted_space_det(_rand((20, d), 8), lambda: h, lambda: sec)
    assert b is vecs and mu is lam

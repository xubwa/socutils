'''_krylov with the implicit subspace matrix (KRYLOV_IMPLICIT_H, no A x kept) against the explicit one, pyscf's
lib.krylov and a direct solve, for many right-hand sides'''
import unittest
import numpy
from pyscf import lib
from pyscf.lib import linalg_helper
from socutils.hessian import rhf_fast


def _system(n=600, nrhs=24, seed=7, scale=.05, symmetric=False):
    rng = numpy.random.default_rng(seed)
    a = rng.standard_normal((n, n)) / numpy.sqrt(n) * scale
    if symmetric:
        a = .5 * (a + a.T)
    a += numpy.diag(rng.uniform(0, .1, n))          # (1 + a) well conditioned (CPHF-like)
    b = rng.standard_normal((nrhs, n))
    return a, b


def _solve(implicit, a, b, hermi=False):
    with lib.temporary_env(rhf_fast, KRYLOV_IMPLICIT_H=implicit):
        rhf_fast._KRYLOV_STATS.clear()
        x = rhf_fast._krylov(lambda v: lib.dot(v, a.T), b, tol=1e-12, max_cycle=200, hermi=hermi)
        return x, dict(rhf_fast._KRYLOV_STATS)


def _direct(a, b):
    return numpy.linalg.solve(numpy.eye(a.shape[0]) + a, b.T).T


class KnownValues(unittest.TestCase):
    def test_nonsymmetric(self):
        a, b = _system()
        x0, s0 = _solve(False, a, b)
        x1, s1 = _solve(True, a, b)
        ref = _direct(a, b)
        self.assertEqual(s0, s1)                     # same iterations and trial vectors
        self.assertLess(abs(x1 - x0).max(), 1e-12)
        self.assertLess(abs(x1 - ref).max(), 1e-7)

    def test_hermi(self):
        a, b = _system(seed=3, symmetric=True)
        x0, _ = _solve(False, a, b, hermi=True)
        x1, _ = _solve(True, a, b, hermi=True)
        self.assertLess(abs(x1 - x0).max(), 1e-12)
        self.assertLess(abs(x1 - _direct(a, b)).max(), 1e-7)

    def test_single_rhs(self):
        a, b = _system(nrhs=1, seed=11)
        x0, _ = _solve(False, a, b[0])
        x1, _ = _solve(True, a, b[0])
        self.assertEqual(x1.shape, b[0].shape)
        self.assertLess(abs(x1 - x0).max(), 1e-12)

    def test_disk(self):
        # the trial vectors in a temporary file, read back in chunks of a few rows
        a, b = _system(seed=9)
        b[:3] *= 1e-6
        x0, s0 = _solve(True, a, b)
        with lib.temporary_env(rhf_fast, KRYLOV_DISK=True, CPHF_DISK_CHUNK=.02):
            x1, s1 = _solve(True, a, b)
        self.assertEqual(s0, s1)
        self.assertLess(abs(x1 - x0).max(), 1e-10)      # chunked Gram-Schmidt: the order of the sums
        # the right-hand sides from a file too (x . b as the trial blocks come)
        f = lib.H5TmpFile()
        f['b'] = b
        with lib.temporary_env(rhf_fast, KRYLOV_DISK=True, CPHF_DISK_CHUNK=.02):
            x2, s2 = _solve(True, a, f['b'])
        f.close()
        self.assertEqual(s0, s2)
        self.assertLess(abs(x2 - x0).max(), 1e-10)

    def test_tiny_rhs(self):
        # right-hand sides 1e-6 of the others: Pople's method (pyscf's too, lindep absolute) stops at a
        # larger error there; the implicit h is no worse than the explicit one
        a, b = _system(seed=5)
        b[:4] *= 1e-6
        ref = _direct(a, b)
        x0 = _solve(False, a, b)[0]
        x1 = _solve(True, a, b)[0]
        # the dropped directions' share of the later <x_k, A x_j> taken along: as the explicit h
        self.assertLess(abs(x1 - x0).max(), 1e-10)
        e0 = abs(x0 - ref).max()
        e1 = abs(x1 - ref).max()
        ep = abs(linalg_helper.krylov(lambda v: lib.dot(v, a.T), b, tol=1e-12, max_cycle=200) - ref).max()
        self.assertAlmostEqual(e0, ep, delta=1e-9 + 1e-6 * ep)
        self.assertLess(e1, 1.5 * e0 + 1e-9)


if __name__ == '__main__':
    unittest.main()

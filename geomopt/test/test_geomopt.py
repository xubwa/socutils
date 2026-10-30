import unittest
import numpy
from pyscf import gto, scf
from socutils import geomopt
from socutils.geomopt import optimizer


class KnownValues(unittest.TestCase):
    def test_rfo_step(self):
        H = numpy.diag([1., 2., -0.5])
        g = numpy.array([.1, -.2, .05])
        dq, lam = optimizer.rfo_step(H, g, 10.)
        self.assertLess(lam, -0.5)
        self.assertAlmostEqual(abs((H - lam * numpy.eye(3)).dot(dq) + g).max(), 0, 12)
        dq, lam = optimizer.rfo_step(H, g, 0.05)
        self.assertAlmostEqual(numpy.linalg.norm(dq), 0.05, 10)

    def test_gediis_quadratic(self):
        # on a quadratic surface GEDIIS reproduces the energies of the points and
        # interpolates to the lowest point of their hull
        rng = numpy.random.RandomState(3)
        A = rng.rand(4, 4)
        A = A.dot(A.T) + numpy.eye(4)
        q = [rng.rand(4) for _ in range(5)]
        E = numpy.array([.5 * x.dot(A).dot(x) for x in q])
        g = [A.dot(x) for x in q]
        c, f = optimizer.gediis_coeffs(E, q, g)
        qi = sum(ci * x for ci, x in zip(c, q))
        self.assertAlmostEqual(c.sum(), 1, 12)
        self.assertGreaterEqual(c.min(), -1e-12)
        self.assertAlmostEqual(f, .5 * qi.dot(A).dot(qi), 10)
        self.assertLessEqual(f, E.min() + 1e-12)

    def test_cubic_min(self):
        # (a - 0.3)^2 (a + 1): minimum at 0.3
        p = lambda a: (a - .3)**2 * (a + 1)
        dp = lambda a: 2 * (a - .3) * (a + 1) + (a - .3)**2
        self.assertAlmostEqual(optimizer.cubic_min(p(0), dp(0), p(1), dp(1)), .3, 12)

    def test_water_hf(self):
        mol = gto.M(atom='O 0 0 0; H 0 0 1.1; H 0 1.05 -.2', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        conv, mol1 = geomopt.optimize(mf)
        self.assertTrue(conv)
        mf1 = scf.RHF(mol1).run(conv_tol=1e-10)
        self.assertLess(abs(mf1.nuc_grad_method().kernel()).max(), 3e-4)
        self.assertAlmostEqual(mf1.e_tot, -74.965901192, 6)

    def test_exact_hessian(self):
        mol = gto.M(atom='C 0 0 0; O 0 0 1.25; H 0 .95 -.6; H 0 -.9 -.55', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        eng = geomopt.PySCFEngine(mf.nuc_grad_method().as_scanner())
        opt = geomopt.Optimizer(eng, hessian='exact', verbose=0)
        conv, x = opt.kernel()
        self.assertTrue(conv)
        self.assertEqual(eng.nhess, 1)
        self.assertLess(opt.ncalc, 8)

    def test_ts_hcn(self):
        # HCN <-> HNC, TS 1 of Baker and Chan at HF/3-21G (-92.24604)
        mol = gto.M(atom='C 0 0 0; N 0 0 1.14838; H 1.58536 0 1.14838', basis='3-21g', verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        conv, mol1 = geomopt.optimize(mf, transition=True)
        self.assertTrue(conv)
        mf1 = scf.RHF(mol1).run(conv_tol=1e-10)
        self.assertAlmostEqual(mf1.e_tot, -92.24604, 4)
        eng = optimizer.PySCFEngine(mf1.nuc_grad_method().as_scanner())
        h = numpy.linalg.eigvalsh(eng.hessian(mol1.atom_coords().ravel()))
        self.assertEqual(int((h < -1e-3).sum()), 1)


if __name__ == '__main__':
    unittest.main()

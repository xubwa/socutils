#!/usr/bin/env python
'''The locking Davidson of socutils.tdscf.lr_locked against pyscf's lr_eigh (two-component TDA) and a dense
Hermitian matrix.'''
import unittest
import numpy
from pyscf import gto
from socutils.dft import dft
from socutils.tdscf import gks, lr_locked


class KnowValues(unittest.TestCase):
    def test_dense(self):
        rng = numpy.random.default_rng(1)
        n = 300
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        a = (a + a.conj().T) * .05 + numpy.diag(numpy.arange(n) * .1 + 1)
        ref = numpy.linalg.eigvalsh(a)[:6]
        d = a.diagonal().real
        nvec = []

        def aop(xs):
            nvec.append(len(xs))
            return xs @ a.T

        def precond(r, e):
            return r / (d - e + 1e-8)
        x0 = numpy.eye(n)[:8]
        conv, e, x = lr_locked.eigh(aop, x0, precond, 6, tol=1e-8, max_cycle=200)
        self.assertTrue(conv.all())
        self.assertLess(abs(e - ref).max(), 1e-12)
        # the batches shrink as roots lock
        self.assertLess(min(nvec[1:]), 6)

    def test_tda_vs_pyscf(self):
        mol = gto.M(atom='Au 0 0 0; H 0 0 1.524', basis='x2c-svpall', verbose=0)
        mf = dft.RKS(mol, xc='pbe0').x2camf()
        mf.conv_tol = 1e-10
        mf.kernel()
        es = {}
        # the reference: pyscf's solver for 12 roots, the lowest 6 (with nstates = 4 it misses the 3.803 eV
        # pair, with 6 the locked solver without buffer roots missed the 3.826 eV root)
        for solver, ns in (('pyscf', 12), ('locked', 6), ('locked', 4)):
            td = gks.TDA(mf)
            td.nstates = ns
            td.conv_tol = 1e-7
            td.davidson = solver
            es[(solver, ns)] = td.kernel()[0]
            self.assertTrue(all(td.converged))
        ref = es[('pyscf', 12)]
        self.assertLess(abs(es[('locked', 6)] - ref[:6]).max(), 1e-9)
        self.assertLess(abs(es[('locked', 4)] - ref[:4]).max(), 1e-9)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python
'''Time-reversal adapted two-component RPA (gks.TDRPA, davidson = 'ssf': real A +- B on the TR-even amplitudes,
lr_locked.rpa_eig) against the paired complex Davidson (davidson = 'paired'), its TDA limit, the dense solver,
spin-free triplet degeneracy against pyscf's restricted TDDFT, and the instability fallback.'''
import unittest
import numpy
from pyscf import gto, dft as pdft
from pyscf.tdscf import rks as prks
from socutils.dft import dft
from socutils.scf import spinor_hf
from socutils.tdscf import gks, lr_locked


def setUpModule():
    global mol, mf
    mol = gto.M(atom='Au 0 0 0; H 0 0 1.524', basis='x2c-svpall', verbose=0)
    mf = dft.RKS(mol, xc='pbe0').x2camf()
    mf.conv_tol = 1e-11
    mf.kernel()


def tearDownModule():
    global mol, mf
    del mol, mf


def _groups(e, tol=1e-6):
    return numpy.cumsum(numpy.r_[0, numpy.diff(e) > tol])


class KnowValues(unittest.TestCase):
    def test_dense(self):
        rng = numpy.random.default_rng(3)
        n = 200
        d = numpy.arange(n) * .02 + 1
        a = rng.standard_normal((n, n))
        a = (a + a.T) * .01 + numpy.diag(d)
        b = rng.standard_normal((n, n))
        b = (b + b.T) * .01

        def precond(r, e):
            return r / (d - e)
        for bb in (b, 0 * b):
            ref = numpy.sort(numpy.sqrt(numpy.linalg.eigvals((a - bb) @ (a + bb)).real))[:6]
            conv, e, P, Q = lr_locked.rpa_eig(lambda p, q: (p @ (a + bb).T, q @ (a - bb).T), numpy.eye(n)[:8],
                                              precond, 6, tol=1e-9, max_cycle=200)
            self.assertTrue(conv.all())
            self.assertLess(abs(e - ref).max(), 1e-11)
            self.assertLess(abs(numpy.einsum('ki,ki->k', P.conj(), Q).real - 1).max(), 1e-12)
        self.assertLess(abs(e - numpy.linalg.eigvalsh(a)[:6]).max(), 1e-11)     # B = 0: the TDA roots
        bb = numpy.diag(d) * 1.2
        with self.assertRaises(lr_locked.RPAInstability):
            lr_locked.rpa_eig(lambda p, q: (p @ (a + bb).T, q @ (a - bb).T), numpy.eye(n)[:8], precond, 6)

    def _solve(self, dav, rijcosx='', ns=6, hsym=False):
        mf.rijcosx = rijcosx
        td = gks.TDRPA(mf)
        td.nstates = ns
        td.conv_tol = 1e-7
        td.davidson = dav
        td.hsym_response = hsym
        e = td.kernel()[0]
        self.assertTrue(all(td.converged))
        f = td.oscillator_strength()
        y2 = numpy.array([numpy.linalg.norm(y) ** 2 for x, y in td.xy])
        mf.rijcosx = ''
        return e, f, y2

    def test_ssf_vs_paired(self):
        # RIJ + COSX: the COSX response K has V(D^H) != V(D)^H at 5e-5 of |V|, the two solvers then differ by ~1e-8
        # Eh; with the Hermitian-symmetrised response (hsym_response) they solve the same RPA
        for rc, hsym, tol in (('', False, 1e-9), ('rijcosx', False, 1e-7), ('rijcosx', True, 1e-9)):
            e0, f0, y0 = self._solve('paired', rc, hsym=hsym)
            e1, f1, y1 = self._solve('ssf', rc, hsym=hsym)
            self.assertLess(abs(e0 - e1).max(), tol, rc)
            if tol > 1e-9:
                continue
            g = _groups(e1)
            self.assertLess(abs(numpy.bincount(g, f0) - numpy.bincount(g, f1)).max(), 1e-7, rc)
            self.assertLess(abs(numpy.bincount(g, y0) - numpy.bincount(g, y1)).max(), 1e-7, rc)

    def test_pbe_missed_root(self):
        '''AuH PBE: the paired solver with nstates = 8 misses the 3.687 eV root (found with nstates = 14); the
        TR-adapted solve with its buffer roots has it'''
        mp = dft.RKS(mol, xc='pbe').x2camf()
        mp.conv_tol = 1e-11
        mp.kernel()
        es = {}
        for dav, ns in (('paired', 14), ('ssf', 8)):
            td = gks.TDRPA(mp)
            td.nstates = ns
            td.conv_tol = 1e-7
            td.davidson = dav
            es[dav] = td.kernel()[0]
        self.assertLess(abs(es['ssf'] - es['paired'][:8]).max(), 1e-9)

    def test_tda_limit(self):
        '''rpa_eig on (A p, A q) from the TR-adapted TDA matvec reproduces the TDA roots'''
        ns = 6
        tda = gks.TDA(mf)
        tda.nstates = ns
        tda.conv_tol = 1e-7
        et = tda.kernel()[0]
        with tda._response_env():
            vind, hd = tda.gen_vind(mf)
            k, (no, nv) = tda._tr_operator()

            def proj(xs):
                z = numpy.asarray(xs).reshape(-1, no, nv)
                return ((z + k(z)) * .5).reshape(len(z), -1)
            ng = 2 * ns + 4
            g = numpy.zeros((ng, hd.size), complex)
            g[numpy.arange(ng), numpy.argsort(hd, kind='stable')[:ng]] = 1

            def aop(ps, qs):
                h = vind(numpy.vstack([ps, qs]))
                return h[:len(ps)], h[len(ps):]
            conv, e, P, Q = lr_locked.rpa_eig(aop, numpy.vstack([g, 1j * g]), lambda r, e: r / (hd - e), ns + 3,
                                              tol=1e-7, project=proj)
        self.assertLess(abs(e[:ns] - et).max(), 1e-9)

    def test_spin_free_triplets(self):
        '''no SOC: the 2c RPA roots are pyscf's restricted singlets once and triplets three times'''
        m = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='6-31g', verbose=0)
        ms = dft.RKS(m, xc='pbe0')
        ms.conv_tol = 1e-11
        ms.kernel()
        td = gks.TDRPA(ms)
        td.nstates = 12
        td.conv_tol = 1e-7
        e = td.kernel()[0]
        self.assertTrue(all(td.converged))
        mr = pdft.RKS(m, xc='pbe0')
        mr.conv_tol = 1e-11
        mr.kernel()
        ref = []
        for singlet, mult in ((True, 1), (False, 3)):
            t = prks.TDDFT(mr)
            t.singlet = singlet
            t.nstates = 8
            t.conv_tol = 1e-8
            ref += list(t.kernel()[0]) * mult
        ref = numpy.sort(ref)[:12]
        self.assertLess(abs(e - ref).max(), 1e-7)

    def test_instability(self):
        '''stretched H2, RHF: the triplet instability; warning and the TDA roots'''
        m = gto.M(atom='H 0 0 0; H 0 0 3.0', basis='cc-pvdz', verbose=0)
        mh = spinor_hf.RHF(m)
        mh.conv_tol = 1e-11
        mh.kernel()
        td = gks.TDRPA(mh)
        td.nstates = 3
        e = td.kernel()[0]
        self.assertTrue(td.unstable)
        tda = gks.TDA(mh)
        tda.nstates = 3
        self.assertLess(abs(e - tda.kernel()[0]).max(), 1e-6)
        td.instability = 'raise'
        with self.assertRaises(lr_locked.RPAInstability):
            td.kernel()


if __name__ == '__main__':
    unittest.main()

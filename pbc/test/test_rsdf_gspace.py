'''RSDF(lr='gspace'): SR-only fitted vectors + exact LR plane-wave rows, with
the G = 0 part of the SR kernel kept out of the vectors.

* the LR rows reproduce the dense G-space long-range exchange exactly;
* the G = 0 term is handled consistently: mean field explicitly, wavefunctions
  as the constant -c0 N (N-1)/2 (a CAS of occupied orbitals = HF; CAHF on
  either backend);
* N2 in a box: with the long range exact, the A/L^3 extrapolation is closer
  to the exact (non-DF) molecular HF than the fit of the whole kernel.
'''
import unittest
import numpy
from pyscf import gto, df, lib
from socutils.scf import spinor_hf
from socutils.mcscf import zcasci
from socutils.pbc import coulomb_reference as cr

try:
    from socutils.pbc.rsdf import RSDF, _libpbcdf
    from socutils.pbc import spinor_scf, pari
    _libpbcdf()
    HAVE_LIB = True
except OSError:
    HAVE_LIB = False


def _quiet(o):
    o.verbose = 0
    return o


@unittest.skipUnless(HAVE_LIB, 'libsocpbcdf.so not built (make)')
class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='N 0 0 -1.04; N 0 0 1.04', unit='B', basis='cc-pvdz', verbose=0)
        cls.aux = df.addons.aug_etb(cls.mol, beta=2.0)

    def test_lr_rows_exact(self):
        cell = cr.molecule_in_box(self.mol, 12.)
        r = RSDF(cell, self.aux, precision=1e-11, nuc_zeta=1e14, omega=.4, lr='gspace')
        r.keep_raw = True
        r.build()
        dm = numpy.random.default_rng(0).normal(size=(cell.nao, cell.nao))
        dm = dm + dm.T
        L = lib.unpack_tril(numpy.asarray(r.cderi[r.naux_kept:]))
        self.assertEqual(L.shape[0], r.nlr)
        klr = numpy.einsum('Pml,ls,Pns->mn', L, dm, L) - r.c_g0 * r.S @ dm @ r.S
        self.assertLess(abs(klr - pari.lr_exchange(r, dm)).max(), 1e-11)
        # SR vectors: the fit of the SR kernel alone
        Ls = lib.unpack_tril(numpy.asarray(r.cderi[:r.naux_kept]))
        eri_fit = numpy.einsum('Pmn,Pls->mnls', Ls, Ls)
        w, v = numpy.linalg.eigh(r.metric_sr)
        keep = w > r.lindep * w.max()
        B = lib.unpack_tril(r.int3c_sr)
        Bw = numpy.einsum('Pq,Pmn->qmn', v[:, keep] / numpy.sqrt(w[keep]), B)
        self.assertLess(abs(eri_fit - numpy.einsum('Pmn,Pls->mnls', Bw, Bw)).max(), 1e-10)

    def test_g0_constant_and_cas(self):
        cell = cr.molecule_in_box(self.mol, 14.)
        res = {}
        for lr, om in (('fit', .5), ('gspace', .3)):
            r = RSDF(cell, self.aux, precision=1e-11, nuc_zeta=1e14, omega=om, lr=lr).build()
            mf = _quiet(spinor_scf.SpinorRSDF(cell, r))
            mf.conv_tol = 1e-11
            e = mf.kernel()
            mc0 = _quiet(zcasci.CASCI(mf, 6, 6))
            mc0.kernel()
            self.assertAlmostEqual(mc0.e_tot, e, 9)
            mc = _quiet(zcasci.CASCI(mf, 12, 6))
            mc.kernel()
            res[lr] = (e, mc.e_tot - e)
        # fit errors only (~1e-6), no G = 0 constant left over
        self.assertLess(abs(res['gspace'][0] - res['fit'][0]), 5e-6)
        self.assertLess(abs(res['gspace'][1] - res['fit'][1]), 5e-7)

    def test_ks_pure_and_hybrid(self):
        '''Pure KS takes the G = 0 term through J only (the J-only DF path),
        hybrids through J and K.'''
        from socutils.pbc import spinor_ks
        cell = cr.molecule_in_box(self.mol, 12.)
        for xc in ('PBE', 'PBE0'):
            e = {}
            for lr, om in (('fit', .5), ('gspace', .3)):
                r = RSDF(cell, self.aux, precision=1e-11, nuc_zeta=1e14, omega=om, lr=lr).build()
                mf = _quiet(spinor_ks.periodic_ks(spinor_hf.SpinorSCF(cell), r, xc=xc))
                mf.grids_level = 3
                e[lr] = mf.kernel()
            self.assertLess(abs(e['gspace'] - e['fit']), 5e-6)

    def test_box_limit_closer_to_exact(self):
        m = _quiet(spinor_hf.SpinorSCF(self.mol))
        e_exact = m.kernel()
        err = {}
        for lr, om in (('fit', .5), ('gspace', .3)):
            res = {}
            for L in (20., 26.):
                cell = cr.molecule_in_box(self.mol, L)
                r = RSDF(cell, self.aux, precision=1e-11, nuc_zeta=1e14, omega=om, lr=lr).build()
                res[L] = _quiet(spinor_scf.SpinorRSDF(cell, r)).kernel()
            (a, ea), (b, eb) = sorted(res.items())
            err[lr] = (eb * b**3 - ea * a**3) / (b**3 - a**3) - e_exact
        # aug-etb DF error of N2 is -4.7e-5; the exact LR removes part of it
        self.assertLess(abs(err['gspace']), abs(err['fit']))
        self.assertLess(abs(err['gspace']), 4.5e-5)


if __name__ == '__main__':
    unittest.main()

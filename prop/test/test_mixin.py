#!/usr/bin/env python
'''Tests of the property mixins (socutils.prop.mixin).

The dipole moment of each mixin against reference values (z component, a.u.;
the molecules lie along z or in the yz plane).  The nonrelativistic and
four-component references agree with pyscf's own dipole of the same density
to 1e-15; the X2CMMF dipole is the DHF one by construction; the spinor and
spin-orbital X2C references agree with each other.  The effective densities
of X2C (somf/test/test_x2c_grad_adjoint.py checks them against get_hfw1)
must be made once per density.
'''

import unittest
from pyscf import gto, scf
from socutils.scf import spinor_hf, ghf
from socutils.somf import x2cmmf
from socutils.prop import mixin


class _RHF(mixin.ScalarPropertyMixin, scf.hf.RHF):
    pass


class _UHF(mixin.ScalarPropertyMixin, scf.uhf.UHF):
    pass


class _DHF(mixin.FourComponentPropertyMixin, scf.dhf.DHF):
    pass


def _run(mf):
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    return mf.run()


def setUpModule():
    global h2o, oh, hbr, x2c_hbr, dhf_h2o
    h2o = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='cc-pvdz', verbose=0)
    oh = gto.M(atom='O 0 0 0; H 0 0 .97', basis='cc-pvdz', spin=1, verbose=0)
    hbr = gto.M(atom='H 0 0 0; Br 0 0 1.41', basis='cc-pvdz', verbose=0)
    x2c_hbr = _run(spinor_hf.SCF(hbr).x2camf())
    dhf_h2o = _run(_DHF(h2o))


class KnownValues(unittest.TestCase):
    def check(self, mu, z):
        self.assertAlmostEqual(abs(mu[:2]).max(), 0, 10)
        self.assertAlmostEqual(mu[2], z, 7)

    def test_scalar(self):
        self.check(_run(_RHF(h2o)).dip_moment(unit='au', verbose=0), 0.80985471422)
        self.check(_run(_UHF(oh)).dip_moment(unit='au', verbose=0), 0.70955289697)

    def test_spin_orbital(self):
        self.check(_run(ghf.GHF(h2o)).dip_moment(unit='au', verbose=0), 0.80985471438)
        self.check(_run(ghf.GHF(hbr).x2camf()).dip_moment(unit='au', verbose=0), -0.43195277501)

    def test_spinor(self):
        self.check(_run(spinor_hf.SCF(h2o)).dip_moment(unit='au', verbose=0), 0.80985471422)
        self.check(x2c_hbr.dip_moment(unit='au', verbose=0), -0.43195277459)
        self.check(x2c_hbr.dip_moment(unit='au', verbose=0, x_response=False), -0.43195277495)

    def test_snso(self):
        # sNSO: the plain picture change of the pyscf helper it is built on
        from socutils.somf import snso
        mf = spinor_hf.SCF(h2o)
        mf.with_x2c = snso.SpinorX2CHelper(h2o)
        mf.with_x2c.boettger = True
        _run(mf)
        self.check(mf.dip_moment(unit='au', verbose=0, x_response=False), 0.80872445045)
        self.assertRaises(NotImplementedError, mf.dip_moment, verbose=0, x_response=True)

    def test_four_component(self):
        self.check(dhf_h2o.dip_moment(unit='au', verbose=0), 0.80846663293)
        # X2CMMF: plain picture change with the decoupling of the DHF Fock
        # matrix, whose orbitals are the DHF ones
        self.check(x2cmmf.x2cmmf(dhf_h2o).dip_moment(unit='au', verbose=0), 0.80846663293)

    def test_cache(self):
        # one adjoint per density and setting
        mf = x2c_hbr
        helper = mf.with_x2c
        mf.__dict__.pop('_prop_cache', None)
        calls = []
        orig = helper.get_hfw1_adjoint

        def counting(*args, **kwargs):
            calls.append(1)
            return orig(*args, **kwargs)
        helper.get_hfw1_adjoint = counting
        try:
            mf.dip_moment(verbose=0)
            mf.dip_moment(verbose=0, origin=(0, 0, 1.))
            mf.expect_int1e('r')
            self.assertEqual(len(calls), 1)
            mf.dip_moment(verbose=0, x_response=False)
            self.assertEqual(len(calls), 2)
            mf.dip_moment(verbose=0, dm=mf.make_rdm1() * .5)
            self.assertEqual(len(calls), 3)
        finally:
            del helper.get_hfw1_adjoint


if __name__ == '__main__':
    print('Tests of the property mixins')
    unittest.main()

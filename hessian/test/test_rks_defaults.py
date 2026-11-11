#!/usr/bin/env python
'''
Regression net for the default fast RKS path

    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='def2-universal-jfit', only_dfj=True)
    mf.grids = socutils.dft.grids.Grids(mol)
    mf = cosx.apply(vxc_fast.apply(mf))
    rhf_fast.RKSHessian(mf).kernel()

against stored numbers of the same model without its approximations (RIJ + exact
K, pyscf grid level 5, the Hessian with every COSX / coarse-grid switch off), at
geometries optimised in that reference model (geomeTRIC, GAU_TIGHT).  The
defaults were once validated on timings and three C/H/N/O organics only; these
cases (third-row atoms, a cation, a soft torsion) fail if

  * the translational sum rule is not imposed on socutils grids (6 - 32 cm-1),
  * the Fock1 / explicit K COSX grid is too coarse (GRIDX0: 5 - 6 cm-1),
  * the CP-SCF's last response stays on the coarse operator grid (CH3SH: 1.9 cm-1),
  * a cached COSX object silently changes its grid (cosx.COSX.build).

    python hessian/test/test_rks_defaults.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto, scf, dft, lib
from pyscf.hessian import thermo
from socutils.hessian import rhf_fast
from socutils.grad import rhf_fast as grad_fast
from socutils.scf import cosx
from socutils.dft import vxc_fast, grids as sgrids

AUX = 'def2-universal-jfit'

# name: (charge, geometry in bohr, E(R1), harmonic frequencies of R1 in cm-1); PBE0 / def2-SVP
CASES = {
    'ch3cl': (0, '''
        C      0.0097395605    -0.0013330374    -0.0039275408
        Cl     3.3643663839     0.0373040261    -0.0009171875
        H     -0.6705111615    -0.0373559807     1.9541356178
        H     -0.6879171397     1.7010992355    -0.9594628236
        H     -0.6488069381    -1.6911962145    -1.0082836547''', -499.72884412,
              (763.74, 1018.45, 1018.46, 1365.49, 1456.23, 1456.23, 3083.36, 3206.55, 3206.58)),
    'ch3sh': (0, '''
        C     -0.0027376440    -0.0063994181     0.0430384142
        S      3.4208415938     0.0293725863    -0.0335368430
        H      3.7889369997    -0.0084672662     2.4902542677
        H     -0.6274545779     0.0198751342    -1.9369096045
        H     -0.7758644923     1.6710024399     0.9925824971
        H     -0.7415894821    -1.7293628381     0.9366427716''', -438.35143568,
              (254.28, 732.43, 804.05, 965.86, 1095.50, 1341.00, 1447.28, 1460.78, 2715.84, 3068.49, 3177.99,
               3186.65)),
    'ch3nh3+': (1, '''
        C      0.0309363148    -0.0107682731    -0.0027153408
        N      2.8503628316    -0.0162515108     0.0009120790
        H     -0.6388801002    -0.0253855644     1.9580113494
        H     -0.6317610038     1.6972698702    -0.9705770180
        H     -0.6383300885    -1.7002916264    -0.9981587979
        H      3.5806673103    -0.0030187638    -1.8015592618
        H      3.5741254505    -1.5867869209     0.8908627909
        H      3.5801676216     1.5368055849     0.9162388095''', -96.02843195,
                (306.14, 913.98, 914.01, 990.12, 1266.14, 1266.14, 1444.84, 1476.13, 1476.13, 1516.86, 1635.43,
                 1635.44, 3105.38, 3233.46, 3233.47, 3423.85, 3510.65, 3510.69)),
}
# max |freq - freq(R1)| of the default path: 0.16 (CH3Cl), 0.15 (CH3SH), 0.34 (CH3NH3+).  With the Fock1 / explicit K
# on GRIDX2 (the default before HESS_COSX_GRID = 3): 0.22 / 0.78 (the 254 cm-1 torsion) / 0.47; without the final
# response grid: 0.44 / 0.93 / 0.39.
FREQ_TOL = 0.5
E_TOL = 2e-5            # E - E(R1): 3.4e-6, 2.9e-6, 1.7e-6
G_TOL = 3e-4            # the gradient at the R1 minimum: 1.2e-4, 9.4e-5, 1.5e-4 (the COSX K gradient on GRIDX2)


def _freq(mol, h):
    v = thermo.harmonic_analysis(mol, h)['freq_wavenumber']
    return numpy.sort(v.real - v.imag)


def _sum_rule_max(h):
    return abs(h.sum(axis=1)).max()


def _default_scf(name):
    charge, atom, e_ref, f_ref = CASES[name]
    mol = gto.M(atom=atom, unit='bohr', basis='def2-svp', charge=charge, verbose=0)
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis=AUX, only_dfj=True)
    mf.grids = sgrids.Grids(mol)
    mf = cosx.apply(vxc_fast.apply(mf))
    mf.kernel()
    return mol, mf, e_ref, numpy.array(f_ref)


class KnownValues(unittest.TestCase):
    def test_sum_rule_auto(self):
        '''sum_rule 'auto': imposed on every DFT grid (socutils and pyscf), not for Hartree-Fock; False gives
        pyscf's Hessian'''
        mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692', basis='def2-svp', verbose=0)
        # socutils grid
        mf = dft.RKS(mol, xc='pbe0')
        mf.grids = sgrids.Grids(mol)
        mf.conv_tol = 1e-11
        mf.kernel()
        hobj = rhf_fast.RKSHessian(mf)
        self.assertEqual(hobj.sum_rule, 'auto')
        h = hobj.kernel()
        raw = rhf_fast.RKSHessian(mf)
        raw.sum_rule = False
        raw = raw.kernel()
        self.assertGreater(_sum_rule_max(raw), 1e-2)     # the fixed-grid Hessian breaks the invariance (0.14 here)
        self.assertLess(_sum_rule_max(h), 1e-4)
        self.assertAlmostEqual(abs(h - sgrids.translational_sum_rule(raw)).max(), 0, 9)
        self.assertGreater(abs(_freq(mol, h) - _freq(mol, raw)).max(), 5.)
        # pyscf grid: imposed too; sum_rule = False is pyscf's Hessian
        mf = dft.RKS(mol, xc='pbe0')
        mf.grids.level = 3
        mf.conv_tol = 1e-11
        mf.kernel()
        ref = mf.Hessian().kernel()
        hobj = rhf_fast.RKSHessian(mf)
        hobj.cpscf_grids = None
        h = hobj.kernel()
        self.assertLess(_sum_rule_max(h), 1e-6)
        self.assertAlmostEqual(abs(h - sgrids.translational_sum_rule(ref)).max(), 0, 7)
        self.assertGreater(abs(h - ref).max(), 1e-6)     # pyscf's own Hessian breaks the invariance
        h2 = rhf_fast.RKSHessian(mf)
        h2.cpscf_grids = None
        h2.sum_rule = False
        self.assertAlmostEqual(abs(h2.kernel() - ref).max(), 0, 7)
        # Hartree-Fock: no grid, nothing imposed
        mf = scf.RHF(mol).run(conv_tol=1e-11)
        h = rhf_fast.RHFHessian(mf)
        h3 = rhf_fast.RHFHessian(mf)
        h3.sum_rule = False
        self.assertAlmostEqual(abs(h.kernel() - h3.kernel()).max(), 0, 12)

    def _check_default(self, name, coarse_fock1=4., final_grid=None, gridx2=None):
        mol, mf, e_ref, f_ref = _default_scf(name)
        self.assertTrue(mf.converged)
        self.assertLess(abs(mf.e_tot - e_ref), E_TOL)
        self.assertLess(abs(grad_fast.Gradients(mf).kernel()).max(), G_TOL)
        h = rhf_fast.RKSHessian(mf).kernel()
        self.assertFalse(numpy.isnan(h).any())
        self.assertLess(_sum_rule_max(h), 1e-4)
        self.assertLess(abs(_freq(mol, h) - f_ref).max(), FREQ_TOL)
        # the faults this net is for
        raw = rhf_fast.RKSHessian(mf)
        raw.sum_rule = False
        self.assertGreater(abs(_freq(mol, raw.kernel()) - f_ref).max(), 5.)
        with lib.temporary_env(rhf_fast, HESS_COSX_GRID=0):
            h0 = rhf_fast.RKSHessian(mf).kernel()
        self.assertGreater(abs(_freq(mol, h0) - f_ref).max(), coarse_fock1)
        if final_grid is not None:
            with lib.temporary_env(rhf_fast, CPHF_COSX_GRID_FINAL=None):
                h0 = rhf_fast.RKSHessian(mf).kernel()
            self.assertGreater(abs(_freq(mol, h0) - f_ref).max(), final_grid)
        if gridx2 is not None:
            with lib.temporary_env(rhf_fast, HESS_COSX_GRID=2):
                h0 = rhf_fast.RKSHessian(mf).kernel()
            self.assertGreater(abs(_freq(mol, h0) - f_ref).max(), gridx2)
        # the COSX objects kept their grids over the gradient and the Hessians (cosx.COSX.build)
        npts = {lev: k.grids.weights.size for lev, k in mf._cosx['k'].items()}
        for lev, n in npts.items():
            self.assertEqual(n, cosx.make_grids(mol, **cosx.GRIDX[lev]).weights.size)
        return mf

    def test_ch3cl(self):
        # third row
        mf = self._check_default('ch3cl')
        # a second kernel() after the gradient / Hessian: the same energy (the final exchange energy on GRIDX3)
        e = mf.e_tot
        self.assertAlmostEqual(mf.kernel(), e, 7)

    def test_ch3sh(self):
        # third row, a soft torsion (254 cm-1): the mode most sensitive to the COSX grids of the Hessian
        self._check_default('ch3sh', final_grid=0.7, gridx2=0.6)       # 0.93, 0.78 (> FREQ_TOL)

    def test_cation(self):
        self._check_default('ch3nh3+')

    def test_exact_k_on_default_grid(self):
        '''RIJ + exact K on the default socutils grid with the sum rule: the XC grid alone (0.2 cm-1 here)'''
        from socutils.scf import md_j
        charge, atom, e_ref, f_ref = CASES['ch3cl']
        mol = gto.M(atom=atom, unit='bohr', basis='def2-svp', charge=charge, verbose=0)
        mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis=AUX, only_dfj=True)
        mf.grids = sgrids.Grids(mol)
        mf = md_j.apply(vxc_fast.apply(mf))
        mf.kernel()
        self.assertLess(abs(mf.e_tot - e_ref), 1e-5)
        h = rhf_fast.RKSHessian(mf).kernel()
        self.assertLess(abs(_freq(mol, h) - numpy.array(f_ref)).max(), 0.5)


if __name__ == '__main__':
    print('Regression tests of the default fast RKS path (socutils grids, COSX, rhf_fast Hessian)')
    unittest.main()

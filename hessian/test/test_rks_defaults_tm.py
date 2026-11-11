#!/usr/bin/env python
'''
The default fast RKS path on a 3d complex, Fe(CO)5 PBE0 / def2-SVP (slower than
test_rks_defaults.py: ~90 s on 4 threads).

Reference: RIJ + exact K on socutils.dft.grids.Grids(intacc=6.0, angular_grid=7)
with the sum rule, at the geometry optimised with RIJ + exact K on the pyscf
level-5 grid (ORCA 6.0.1 RIJCOSX DefGrid3 is within 0.43 cm-1 of it).

Findings of the validation (2026-11-10), in cm-1 over the modes above 100:

    all defaults                                                  16.6
    all COSX on GRIDX3, CP-SCF XC response on the SCF grid         0.6
    back to the defaults one at a time from there:
      SCF COSX on GRIDX2                                          +2.8
      CP-SCF COSX operator on GRIDX0, last response on GRIDX2     +3.2
      Fock1 / explicit K on GRIDX2                                +2.8
      CP-SCF XC grid (angular_grid 2, intacc 3.467)               +9.9

Every one of these grids uses the five-region Lebedev table 2 (or coarser): 26
points in the region of the 3d shell.  Table 3 (50 points there) is enough at
any radial accuracy tried (CPSCF_GRID_ARGS angular_grid=3: 0.04; CP-SCF COSX
grid intacc 3.0 / angular_grid 3: 0.01; SCF COSX grid intacc 3.816 /
angular_grid 3: 0.01), table 2 fails at any (intacc 4.338: 2.8).

test_defaults is an expected failure until the defaults handle 3d atoms; when
it starts to pass ("unexpected success"), remove the decorator.

    python hessian/test/test_rks_defaults_tm.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto, dft, lib
from pyscf.hessian import thermo
from socutils.hessian import rhf_fast
from socutils.scf import cosx
from socutils.dft import vxc_fast, grids as sgrids

FECO5 = '''
    Fe     0.0137629860    -0.0119878582    -0.0020400486
    C      0.0157134027    -0.0118739538     3.3979503586
    O      0.0169211331    -0.0118126249     5.5513988832
    C      0.0116016777    -0.0122555619    -3.4020276170
    O      0.0102043538    -0.0124311554    -5.5554752118
    C      3.3960219350    -0.0315974288    -0.0040724355
    O      5.5561770736    -0.0441173841    -0.0053662126
    C     -1.6601217224     2.9270889603    -0.0011881449
    O     -2.7291472848     4.8042204028    -0.0006505736
    C     -1.6943217204    -2.9313137345    -0.0008596012
    O     -2.7852141570    -4.7958230942    -0.0001259598
'''
E_REF = -1828.99675450      # RIJ + exact K, pyscf grid level 5
F_REF = numpy.array((
    50.59, 50.63, 101.95, 101.96, 109.47, 109.60, 114.18, 385.06, 385.08, 386.92, 438.51, 459.65, 459.66, 465.72,
    492.46, 508.34, 508.39, 586.85, 586.87, 650.19, 688.44, 688.47, 2144.05, 2144.64, 2168.74, 2169.40, 2246.87))
HI = F_REF > 100


def _freq(mol, h):
    v = thermo.harmonic_analysis(mol, h)['freq_wavenumber']
    return numpy.sort(v.real - v.imag)


def _scf(mol, **kwargs):
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='def2-universal-jfit', only_dfj=True)
    mf.grids = sgrids.Grids(mol)
    mf = cosx.apply(vxc_fast.apply(mf), **kwargs)
    mf.kernel()
    return mf


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom=FECO5, unit='bohr', basis='def2-svp', verbose=0)

    @unittest.expectedFailure
    def test_defaults(self):
        mol = self.mol
        mf = _scf(mol)
        self.assertTrue(mf.converged)
        h = rhf_fast.RKSHessian(mf).kernel()
        self.assertLess(abs(_freq(mol, h) - F_REF)[HI].max(), 1.5)      # 16.6

    def test_gridx3(self):
        '''conservative settings: SCF COSX and Fock1 / explicit K on GRIDX3, the CP-SCF grids (COSX operator, XC
        response) with the Lebedev table 3 (as good as every COSX grid GRIDX3 and the XC response on the SCF grid)'''
        mol = self.mol
        mf = _scf(mol, grids=3)
        self.assertTrue(mf.converged)
        self.assertLess(abs(mf.e_tot - E_REF), 5e-5)                    # -2.5e-5 (the default XC grid: -3.6e-5)
        hobj = rhf_fast.RKSHessian(mf)
        with lib.temporary_env(rhf_fast, CPHF_COSX_GRID=dict(intacc=3.0, angular_grid=3), CPHF_COSX_GRID_FINAL=3,
                               HESS_COSX_GRID=3, CPSCF_GRID_ARGS=dict(angular_grid=3, intacc=3.467)):
            h = hobj.kernel()
        self.assertFalse(numpy.isnan(h).any())
        self.assertLess(abs(_freq(mol, h) - F_REF)[HI].max(), 1.0)      # 0.6


if __name__ == '__main__':
    print('Default fast RKS path on Fe(CO)5')
    unittest.main()

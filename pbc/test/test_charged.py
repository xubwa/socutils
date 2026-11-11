'''Charged-cell correction (socutils.pbc.charged): He+ in cubic boxes with the X2CAMF GHF of socutils.pbc.amf.
After the q^2 nu_M / 2 correction the charged cell has the finite-size error of the neutral one.'''
import unittest
import numpy as np
from pyscf import gto as mgto, scf as mscf
from pyscf.pbc import gto, scf
from pyscf.pbc.x2c import x2c1e as pbc_x2c1e
from socutils.pbc.amf import PBCSpinOrbitalX2CAMFHelper
from socutils.pbc import charged
from socutils.scf.x2camf_hf import x2camf_ghf


def _periodic(q, L, basis='sto-3g'):
    cell = gto.Cell(atom='He 0 0 0', a=np.eye(3) * L, unit='B', basis=basis, charge=q, spin=q % 2, verbose=0)
    cell.build()
    mf = pbc_x2c1e.x2c1e_gscf(scf.GHF(cell))
    mf.with_x2c = PBCSpinOrbitalX2CAMFHelper(cell, with_gaunt=True, with_breit=True)
    mf.verbose = 0
    charged.correct(mf)
    return mf.kernel(), mf


def _molecular(q, basis='sto-3g'):
    mol = mgto.M(atom='He 0 0 0', basis=basis, charge=q, spin=q % 2, verbose=0)
    return x2camf_ghf(mscf.GHF(mol)).kernel()


class KnownValues(unittest.TestCase):
    def test_correction_value(self):
        cell = gto.Cell(atom='He 0 0 0', a=np.eye(3) * 10., unit='B', basis='sto-3g', charge=1, spin=1,
                        verbose=0).build()
        # simple cubic Madelung constant 2.837297..., nu_M = alpha / L
        self.assertAlmostEqual(charged.charge_correction(cell), .5 * 2.8372974794806 / 10., 6)
        neutral = gto.Cell(atom='He 0 0 0', a=np.eye(3) * 10., unit='B', basis='sto-3g', verbose=0).build()
        self.assertEqual(charged.charge_correction(neutral), 0.)

    def test_he_cation(self):
        e1, mf = _periodic(1, 15.)
        e0, _ = _periodic(0, 15.)
        err1 = e1 - _molecular(1)
        err0 = e0 - _molecular(0)
        # uncorrected the cation is off by -q^2 nu_M / 2 (-0.0946 Eh here); corrected, by the neutral error
        self.assertAlmostEqual(err1, err0, 5)
        self.assertGreater(mf.charge_correction, .09)


if __name__ == '__main__':
    unittest.main()

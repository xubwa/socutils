'''scf.addons.projected_frac_occ: open shell chosen by character (weight in a
basis-function subspace), the valence shell among the candidates by energy.'''
import unittest
import numpy
from pyscf import gto
from socutils.scf import spinor_hf, addons


def _run(mol, *args):
    mf = spinor_hf.SpinorSCF(mol)
    mf.verbose = 0
    if args:
        addons.projected_frac_occ(mf, *args)
    return mf, mf.kernel()


class KnownValues(unittest.TestCase):
    def test_full_shell_is_closed_shell(self):
        # the p span also holds the virtual 3p/4p (weight ~1): 2p must be picked
        mol = gto.M(atom='Ne 0 0 0', basis='cc-pvtz', verbose=0)
        e0 = _run(mol)[1]
        mf, e1 = _run(mol, addons.spinor_ao_indices(mol, 0, l=1), 6, 6)
        self.assertAlmostEqual(e1, e0, 10)
        self.assertTrue((mf.mo_energy[mf.open_shell_mo] < 0).all())

    def test_open_p_shell(self):
        mol = gto.M(atom='O 0 0 0', basis='cc-pvtz', verbose=0, spin=2)
        mf, e = _run(mol, addons.spinor_ao_indices(mol, 0, l=1), 6, 4)
        o = mf.open_shell_mo
        self.assertTrue(numpy.allclose(mf.mo_occ[o], 4 / 6))
        self.assertAlmostEqual(mf.mo_occ.sum(), mol.nelectron, 12)
        self.assertLess(numpy.ptp(mf.mo_energy[o]), 1e-6)      # degenerate 2p
        mo, ncore = addons.cas_order(mf)
        self.assertEqual(ncore, 4)
        self.assertLess(abs(mo[:, ncore:ncore + 6] - mf.mo_coeff[:, numpy.sort(o)]).max(), 1e-12)


if __name__ == '__main__':
    unittest.main()

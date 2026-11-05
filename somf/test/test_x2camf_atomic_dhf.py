#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
The atomic four-component Dirac-HF solved inside the x2camf library against
pyscf's molecular four-component DHF on the same atom.

The atomic code averages open shells over all spinors of an l shell
(fractional occupation n_l / (2(2l+1)) for every spinor of the shell,
irrespective of j), which gives a spherical density.  pyscf.scf.dhf.DHF
reproduces this with a get_occ that assigns the same occupations; note that
pyscf.scf.addons.frac_occ is a different scheme (it splits by orbital
energy, i.e. by j).  Same uncontracted basis, point nucleus, speed of light.

Fluorine (1s2 2s2 2p5): the energies agree to the SCF convergence, for the
Dirac-Coulomb, Dirac-Coulomb-Gaunt and Dirac-Coulomb-Breit Hamiltonians.
'''

import unittest
import numpy
from pyscf import gto, lib
from pyscf.scf import dhf
import socutils   # noqa: F401  (puts the bundled x2camf first on sys.path)
from x2camf import libx2camf


def setUpModule():
    global mol, shell, exp_a
    basis = gto.mole.uncontracted_basis(gto.load('cc-pvdz', 'F'))
    mol = gto.M(atom='F 0 0 0', basis={'F': basis}, spin=1, verbose=0)
    shell = numpy.array([b[0] for b in basis])
    exp_a = numpy.array([b[-1][0] for b in basis])


def tearDownModule():
    global mol, shell, exp_a
    del mol, shell, exp_a


def atomic_dhf_energy(with_gaunt, with_breit):
    '''Energy of the atomic DHF of the x2camf library, from the returned
    four-component h, Fock and density matrices: E = 1/2 tr D (h + F).'''
    flavor = with_gaunt + 2 * with_breit
    res = libx2camf.atm_integrals(flavor, 9, shell[-1] + 1, len(shell), 0, shell, exp_a,
                                  speed_of_light=lib.param.LIGHT_SPEED)
    h, fock, dm = (numpy.asarray(res[i]) for i in (2, 3, 11))
    return .5 * numpy.einsum('ij,ji->', dm, h + fock).real


def pyscf_dhf_energy(with_gaunt, with_breit):
    mf = dhf.DHF(mol)
    mf.with_gaunt = with_gaunt
    mf.with_breit = with_breit
    mf.conv_tol = 1e-11
    n2c = mol.nao_2c()

    def get_occ(mo_energy=None, mo_coeff=None):
        # the four 1s, 2s spinors occupied, 2p^5 spread evenly over the six 2p spinors
        occ = numpy.zeros(mo_energy.size)
        occ[n2c:n2c + 4] = 1
        occ[n2c + 4:n2c + 10] = 5. / 6
        return occ
    mf.get_occ = get_occ
    e = mf.kernel()
    assert mf.converged
    return e


class KnownValues(unittest.TestCase):
    def test_dirac_coulomb(self):
        e_atomic = atomic_dhf_energy(False, False)
        self.assertAlmostEqual(e_atomic, -99.1185454116, 8)
        self.assertAlmostEqual(pyscf_dhf_energy(False, False), e_atomic, 8)

    def test_dirac_coulomb_gaunt(self):
        e_atomic = atomic_dhf_energy(True, False)
        self.assertAlmostEqual(e_atomic, -99.1067246342, 8)
        self.assertAlmostEqual(pyscf_dhf_energy(True, False), e_atomic, 8)

    def test_dirac_coulomb_breit(self):
        e_atomic = atomic_dhf_energy(True, True)
        self.assertAlmostEqual(e_atomic, -99.1072348472, 8)
        self.assertAlmostEqual(pyscf_dhf_energy(True, True), e_atomic, 8)


if __name__ == '__main__':
    print('Atomic DHF of the x2camf library vs pyscf 4c DHF')
    unittest.main()

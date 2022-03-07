'''
Fraction-occupation Dirac Hartree-Fock.
'''
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, dhf

zquatev = None
try:
    import zquatev
except ImportError:
    pass

class FRAC_RDHF(dhf.RDHF):
    nopen = None
    nact = None
    def __init__(self, mol, nopen=0, nact=0):
        # don't require electron number to be even since fraction occ is allowed.
        if zquatev is None:
            raise RuntimeError('zquatev library is required to perform Kramers-restricted DHF')
        dhf.UHF.__init__(self, mol)
        self.nopen = nopen
        self.nact = nact

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        nopen = self.nopen
        nact = self.nact
        nclose = mol.nelectron - nact
        c = lib.param.LIGHT_SPEED
        n4c = len(mo_energy)
        n2c = n4c // 2
        mo_occ = numpy.zeros(n2c * 2)
        if mo_energy[n2c] > -1.999 * c**2:
            if nopen == 0:
                mo_occ[n2c:n2c+mol.nelectron] = 1
            else:
                mo_occ[n2c:n2c+nclose] = 1
                mo_occ[n2c+nclose:n2c+nclose+nopen] = 1.*nact/nopen
        else:
            lumo = mo_energy[mo_energy > -1.999 * c**2][mol.nelectron]
            mo_occ[mo_energy > -1.999 * c**2] = 1
            mo_occ[mo_energy >= lumo] = 0
        if self.verbose >= logger.INFO:
            if nopen == 0:
                homo_ndx = mol.nelectron
            else:
                homo_ndx = nclose + nopen
            logger.info(self, 'HOMO %d = %.12g  LUMO %d = %.12g',
                            n2c+homo_ndx, mo_energy[n2c+homo_ndx-1],
                            n2c+homo_ndx+1, mo_energy[n2c+homo_ndx])
            logger.debug1(self, 'NES  mo_energy = %s', mo_energy[:n2c])
            logger.debug(self, 'PES  mo_energy = %s', mo_energy[n2c:])
        return mo_occ

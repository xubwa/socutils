# refactor the spinor hf class
# the idea is to only work out the diagonalization under spinor basis.

from pyscf import gto, lib
from pyscf.scf import hf, ghf
import numpy as np
from functools import reduce
from pyscf.lib import einsum
from pyscf.socutils import spinor_hf

def eig(obj, h, s):
    mol = obj.mol
    c2 = np.vstack(mol.sph2spinor_coeff())
    h_spinor = einsum('ip,pq,qj->ij', c2.T.conj(), h,c2)
    s_spinor = einsum('ip,pq,qj->ij', c2.T.conj(), s,c2)
    e, c_spinor = obj._eigh(h_spinor, s_spinor)
    c = np.dot(c2, c_spinor)
    return e, c

class SpinorSCF(ghf.GHF):
    def eig(self, h, s):
        return eig(self, h, s)
    
class SymmSpinorSCF(SpinorSCF):
    def __init__(self, mol, symmetry=None, occup=None):
        SpinorSCF.__init__(self, mol)
        if symmetry is 'linear' or symmetry is 'sph':
            self.irrep_ao = spinor_hf.symmetry_label(mol, symmetry)
            self.irrep_mo = None
        else:
            raise NotImplementedError
        
        self.occupation = occup
        
    def eig(self, h, s):
        mol = self.mol
        c2 = np.vstack(mol.sph2spinor_coeff())
        h_spinor = einsum('ip,pq,qj->ij', c2.T.conj(), h,c2)
        s_spinor = einsum('ip,pq,qj->ij', c2.T.conj(), s,c2)
        e, c_spinor = spinor_hf.eig(self, h_spinor, s_spinor, irrep=self.irrep_ao)
        c = lib.tag_array(np.dot(c2, c_spinor),ang_tag=c_spinor.ang_tag)
        self.irrep_mo = e.irrep_tag
        return e, c
    
    def get_occ(self, mo_energy=None, mo_coeff=None):
        if self.occupation is None:
            return SpinorSCF.get_occ(self, mo_energy, mo_coeff)
        elif self.irrep_mo is None:
            return SpinorSCF.get_occ(self, mo_energy, mo_coeff)
        else:
            return spinor_hf.get_occ_symm(self, self.irrep_ao, self.occupation, self.irrep_mo, mo_energy, mo_coeff)

JHF = SpinorSCF
SymmJHF = SymmSpinorSCF


from pyscf.dft import gks, r_numint
class SpinorKS(gks.GKS):
    def eig(self, h, s):
        return eig(self, h, s)


if __name__ == '__main__':
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = None

    mol.atom = [['Ne', (0, 0, 0)], ]
    mol.basis = 'ccpvdz'
    mol.build(0, 0)

##############
# SCF result
    mf = SpinorSCF(mol).x2c1e()
    #method.init_guess = '1e'
    mf.kernel()

    ks = SpinorKS(mol).x2c1e()
    ks.kernel()

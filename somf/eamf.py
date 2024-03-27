import x2camf
import numpy as np
from functools import reduce
from pyscf import x2c
from pyscf.data import elements
from socutils import somf
from socutils.somf import x2c_grad
from socutils.scf import spinor_hf
from pyscf import gto, scf
import x2camf
from x2camf import libx2camf
def x2c1e_hfw0_4cmat(h4c, m4c):
    n4c = h4c.shape[0]
    n2c = n4c//2
    hLL = h4c[:n2c,:n2c]
    hLS = h4c[:n2c,n2c:]
    hSL = h4c[n2c:,:n2c]
    hSS = h4c[n2c:,n2c:]
    sLL = m4c[:n2c,:n2c]
    sSS = m4c[n2c:,n2c:]

    _, _, x, st, r, h2c, _, _ = x2c_grad.x2c1e_hfw0_block(hLL, hSL, hLS, hSS, sLL, sSS)
    return x, st, r, h2c

def to_2c(x, r, h4c):
    n4c = h4c.shape[0]
    n2c = n4c//2
    hLL = h4c[:n2c,:n2c]
    hLS = h4c[:n2c,n2c:]
    hSL = h4c[n2c:,:n2c]
    hSS = h4c[n2c:,n2c:]

    l = hLL + np.dot(hLS, x) + np.dot(hLS, x).T.conj() + reduce(np.dot, (x.T.conj(), hSS, x))
    h2c = reduce(np.dot, (r.T.conj(), l, r))
    return h2c

def eamf(x2cobj, verbose=None, gaunt=False, breit=False, pcc=True, aoc=False, nucmod=None):
    mol = x2cobj.mol
    if nucmod is None:
        nucmod = mol.nucmod
    soc_int_flavor = 0
    print(gaunt, breit,aoc,pcc,nucmod)
    soc_int_flavor += gaunt << 0
    soc_int_flavor += breit << 1
    soc_int_flavor += nucmod << 2 
    soc_int_flavor += aoc << 3
    soc_int_flavor += False << 4 # this parameter for spin dependant gaunt

    uniq_atoms = set([a[0] for a in mol._atom])
    den_2c = {}
    den_4c = {}
    for atom in uniq_atoms:
        symbol = gto.mole._std_symbol(atom)
        atom_number = elements.charge(symbol)
        raw_bas = gto.mole.uncontracted_basis(mol._basis[atom])
        #amf_internal_basis
        shell = []
        exp_a = []
        for bas in raw_bas:
            shell.append(bas[0])
            exp_a.append(bas[-1][0])
        shell = np.asarray(shell)
        exp_a = np.asarray(exp_a)
        nbas = shell.shape[0]
        nshell = shell[-1] + 1
        so_2c, fock_2c, density_2c, atom_X, atom_R, so_4c, fock_4c, density_4c = x2camf.libx2camf.atm_integrals(soc_int_flavor, atom_number, nshell, nbas, verbose, shell, exp_a)
        den_4c[atom] = density_4c
        den_2c[atom] = density_2c

    xmol, _  = x2cobj.get_xmol()
    n2c = xmol.nao_2c()
    atom_slices = xmol.aoslice_2c_by_atom()
    density_4c = x2camf.x2camf.construct_molecular_matrix(den_4c, atom_slices, xmol, n2c, True)
    density_2c = x2camf.x2camf.construct_molecular_matrix(den_2c, atom_slices, xmol, n2c, False)
    mf_4c = scf.DHF(xmol)
    mf_4c.with_gaunt = gaunt
    mf_4c.with_breit = breit
    h1e_4c = mf_4c.get_hcore()
    s4c = mf_4c.get_ovlp()
    veff_4c = mf_4c.get_veff(dm=density_4c)
    fock_4c = h1e_4c + veff_4c
    x, st, r, h2c = x2c1e_hfw0_4cmat(fock_4c, s4c)
    mf_2c = scf.X2C(xmol)
    veff_2c = mf_2c.get_veff(dm=density_2c)
    heff = to_2c(x, r, fock_4c) - veff_2c
    return heff   

class SpinorEAMFX2CHelper(x2c.x2c.SpinorX2CHelper):
    def __init__(self, mol, with_gaunt=False, with_breit=False, with_pcc=True, with_aoc=False):
        super().__init__(mol)
        self.gaunt = with_gaunt
        self.breit = with_breit
        self.pcc = with_pcc
        self.aoc = with_aoc
        if gto.mole._parse_nuc_mod(self.mol.nucmod) == 2:
            self.gau_nuc = True
        else:
            self.gau_nuc = False
        
    def eamf(self):
        print(self.gaunt,self.breit,self.pcc,self.aoc)
        return eamf(self, self.mol.verbose, self.gaunt, self.breit, self.pcc, self.aoc, self.gau_nuc)
    
    def get_hcore(self, mol):
        return self.eamf()
    
if __name__ == '__main__':
    mol = gto.M(atom = 'Ne 0 0 0; Ar 0 0 1.8', basis='uncccpvdz', verbose=5)
    x2cobj = SpinorEAMFX2CHelper(mol, with_gaunt=True, with_breit=True)
    from socutils.scf import x2camf_hf
    mf = x2camf_hf.SCF(mol)
    mf.with_x2c=x2cobj
    mf.kernel()
    print(mf.mo_energy)

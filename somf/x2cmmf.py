#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
X2CMMF with a reevaluation of Fock matrix
'''
import x2camf
import numpy as np
import scipy
from functools import reduce
from pyscf import x2c
from pyscf.x2c.x2c import _decontract_spinor
from pyscf.lib import chkfile
from pyscf.data import elements
from socutils import somf
from socutils.somf import x2c_grad
from socutils.scf import spinor_hf
from socutils.tools import spinor2sph
from pyscf import gto, scf

def x2c1e_hfw0_4cmat(h4c, m4c, mol=None):
    n4c = h4c.shape[0]
    n2c = n4c//2
    hLL = h4c[:n2c,:n2c]
    hLS = h4c[:n2c,n2c:]
    hSL = h4c[n2c:,:n2c]
    hSS = h4c[n2c:,n2c:]
    sLL = m4c[:n2c,:n2c]
    sSS = m4c[n2c:,n2c:]

    a, e, x, st, r, h2c, _, _ = x2c_grad.x2c1e_hfw0_block(hLL, hSL, hLS, hSS, sLL, sSS)
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
    atm_ints = {}

    for atom in uniq_atoms:
        symbol = gto.mole._std_symbol(atom)
        atom_number = elements.charge(symbol)
        raw_bas = gto.mole.uncontracted_basis(mol._basis[atom])
        shell = []
        exp_a = []
        for bas in raw_bas:
            shell.append(bas[0])
            exp_a.append(bas[-1][0])
        shell = np.array(shell)
        exp_a = np.array(exp_a)
        nbas = shell.shape[0]
        nshell = shell[-1] + 1
        atm_ints[atom] = x2camf.libx2camf.atm_integrals(soc_int_flavor, atom_number, nshell, nbas, verbose, shell, exp_a)
    x2cobj.atomic_integrals = atm_ints
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};

    xmol, _  = x2cobj.get_xmol()
    n2c = xmol.nao_2c()
    atom_slices = xmol.aoslice_2c_by_atom()
    mf_4c = scf.DHF(xmol)
    mf_4c.with_gaunt = gaunt
    mf_4c.with_breit = breit
    h1e_4c = mf_4c.get_hcore()
    s4c = mf_4c.get_ovlp()
    print('amf_type', x2cobj.amf_type)
    x2cobj.h4c = h1e_4c
    x2cobj.m4c = s4c
    x2cobj.soc_matrix = np.zeros((n2c, n2c))
    density_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 12), atom_slices, xmol, n2c, False)
    x2cobj.density_2c = density_2c
    if x2cobj.amf_type == 'eamf':
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};
        density_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 11), atom_slices, xmol, n2c, True)
        density_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 12), atom_slices, xmol, n2c, False)
        atomic_h1e_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 2), atom_slices, xmol, n2c, True)
        atomic_fock_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 3), atom_slices, xmol, n2c, True)
        atomic_fock_4c_2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 5), atom_slices, xmol, n2c, True)
        atomic_fock_2c_2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 6), atom_slices, xmol, n2c, False)
        atm_x = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        atm_r = construct_molecular_matrix(extract_ith_integral(atm_ints, 1), atom_slices, xmol, n2c, False)
        atomic_fock_2c2e_fw = to_2c(atm_x, atm_r, atomic_fock_4c_2e)
        density_4c_ss = density_4c.copy()
        ene_corr = np.einsum('ij,ji->', atomic_fock_4c_2e, density_4c).real-np.einsum('ij,ji->',atomic_fock_2c_2e, density_2c).real
        ene_corr2 = np.einsum('ij,ji->',atomic_fock_2c2e_fw - atomic_fock_2c_2e, density_2c).real
        print(ene_corr*0.5, ene_corr2*0.5)
        vj_4c, vk_4c= mf_4c.get_jk(dm=density_4c)
        veff_4c = vj_4c - vk_4c
        fock_4c = h1e_4c + veff_4c
        x, st, r, h2c = x2c1e_hfw0_4cmat(fock_4c, s4c, xmol)
        mf_2c = spinor_hf.SCF(xmol)
        veff_2c = mf_2c.get_veff(dm=density_2c)
        x2cobj.veff_2c=veff_2c
        h_x2c = to_2c(x, r, fock_4c) - veff_2c
        x2cobj.h4c = h1e_4c + veff_4c
        x2cobj.soc_matrix = h_x2c - to_2c(x, r, h1e_4c)
    h_x2c = reduce(np.dot, (contr_coeff.T.conj(), h_x2c, contr_coeff))
    return h_x2c

class SpinorX2CMMFHelper(x2c.x2c.SpinorX2CHelper):
    hcore = None
    def __init__(self, mol, dchf):
        super().__init__(mol)
        self.mf4c = dchf
        self.vj4c, self.vk4c = dchf.get_jk(dm=dchf.make_rdm1())
        self.h4c = self.vj4c-self.vk4c + dchf.get_hcore()
        self.m4c = dchf.get_ovlp()
        self.veff_2c = None
        self.soc_matrix = None
        
    def mmf(self):
        vj_4c, vk_4c= self.mf4c.get_jk()
        veff_4c = vj_4c - vk_4c
        fock_4c = self.h4c
        h1e_4c = fock_4c - veff_4c
        x, st, r, h2c = x2c1e_hfw0_4cmat(fock_4c, self.m4c, self.mol)
        occ_4c = self.mf4c.get_occ()
        n4c = occ_4c.shape[0]
        n2c = n4c // 2
        mo_coeff = self.mf4c.mo_coeff
        rinv = np.linalg.inv(r)
        mo_l = mo_coeff[:n2c,n2c:]
        mo_x2c = np.dot(rinv, mo_l)
        mf_2c = spinor_hf.SCF(self.mol)
        occ_2c = occ_4c[n2c:]
        dm2c = mf_2c.make_rdm1(mo_coeff=mo_x2c, mo_occ=occ_2c)
        veff_2c = mf_2c.get_veff(dm=dm2c)
        self.veff_2c=veff_2c
        h_x2c = to_2c(x, r, fock_4c) - veff_2c
        self.soc_matrix = h_x2c - to_2c(x, r, h1e_4c)
        return h_x2c

    
    def get_hcore(self, mol, screen=False):
        if self.hcore is None:
            self.hcore = self.mmf()
        return self.hcore

    def save_hcore(self, filename='mmf.chk'):
        if self.hcore is None:
            chkfile.dump(filename, 'mmf_integral', self.eamf())
        else:
            chkfile.dump(filename, 'mmf_integral', self.hcore())
        if self.h4c is not None:
            chkfile.dump(filename, 'h4c', self.h4c)
        if self.m4c is not None:
            chkfile.dump(filename, 'm4c', self.m4c)
        if self.soc_matrix is not None:
            chkfile.dump(filename, 'soc_integral', self.soc_matrix)

    def load_hcore(self, filename='mmf.chk'):
        try:
            self.hcore = chkfile.load(filename, 'mmf_integral')
        except:
            raise ValueError('No mmf integral found in the chkfile')
        try:
            self.h4c = chkfile.load(filename, 'h4c')
        except:
            self.h4c = None
        try:
            self.m4c = chkfile.load(filename, 'm4c')
        except:
            self.m4c = None
        try:
            self.soc_matrix = chkfile.load(filename, 'soc_integral')
        except:
            self.soc_matrix = None

    def get_soc_integrals(self):
        return self.soc_matrix

    def get_hfw1(self, h4c1, s4c1=None, x_response=True):
        if self.h4c is None:
            self.get_hcore(self.mol)
        n4c = self.h4c.shape[0]
        n2c = n4c//2
        hLL = self.h4c[:n2c,:n2c]
        hLS = self.h4c[:n2c,n2c:]
        hSL = self.h4c[n2c:,:n2c]
        hSS = self.h4c[n2c:,n2c:]
        sLL = self.m4c[:n2c,:n2c]
        sSS = self.m4c[n2c:,n2c:]

        a, e, x, st, r, l, h4c, m4c = x2c_grad.x2c1e_hfw0_block(hLL, hSL, hLS, hSS, sLL, sSS)
        if x_response is True:
            hfw1 = x2c_grad.get_hfw1(a, x, st, m4c, h4c, e, r, l, h4c1, s4c1)
        else:
            hfw1 = to_2c(x, r, h4c1)
        return hfw1
    
    def get_xr(self):
        if self.h4c is None:
            self.get_hcore(self.mol)
        h4c = self.h4c
        s4c = self.m4c
        xmol = self.get_xmol()
        x, st, r, h2c = x2c1e_hfw0_4cmat(h4c, s4c, xmol)
        return x, r

    def to_4c_coeff(self, c2c):
        x, r = self.get_xr()
        cl = np.dot(r, c2c)
        cs = np.dot(x, cl)
        n2c = c2c.shape[0]
        dtype = c2c.dtype
        c4c = np.zeros((n2c*2,n2c*2), dtype=dtype)
        c4c[:n2c, n2c:] = cl
        c4c[n2c:, n2c:] = cs
        return c4c


if __name__ == '__main__':
    mol = gto.M(atom = 'Ne 0 0 0; Ar 0 0 1.8', basis='uncccpvdz', verbose=5)
    x2cobj = SpinorEAMFX2CHelper(mol, with_gaunt=True, with_breit=True)
    from socutils.scf import x2camf_hf
    mf = x2camf_hf.SCF(mol)
    mf.with_x2c=x2cobj
    mf.kernel()
    print(mf.mo_energy)

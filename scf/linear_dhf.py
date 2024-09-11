from functools import reduce
import numpy as np
import scipy
import re
from pyscf import lib
from pyscf.scf import dhf
from socutils.scf.spinor_hf import symmetry_label

def eig(mf, h, s, irrep=None):
    print('symmetry adapted eigen value')
    mol = mf.mol

    if irrep is None:
        raise ValueError("An irrep is desired as input")
    cs = []
    es = []
    n2c = mf.mol.nao_2c()
    n4c = n2c * 2
    e = np.zeros(n4c)
    c = np.zeros((n4c, n4c), dtype=complex)
    irrep_tag = np.empty(n4c, dtype='U10')
    for ir in irrep:
        ir_idx = irrep[ir]
        ir_idx = np.hstack((ir_idx, ir_idx+n2c))
        hi = h[np.ix_(ir_idx, ir_idx)]
        si = s[np.ix_(ir_idx, ir_idx)]
        ei, ci = mf._eigh(h[np.ix_(ir_idx,ir_idx)], s[np.ix_(ir_idx,ir_idx)])
        c[np.ix_(ir_idx,ir_idx)] = ci
        e[ir_idx] = ei
        irrep_tag[ir_idx] = str(ir)
        angular_tag = None
    # process max contributing ao
    # temporarily deprecate this feature
    # from scipy.linalg import sqrtm
    # s_sqrtm = sqrtm(s)
    # c_tilde = np.dot(s_sqrtm, c)
    # labels = np.array(mol.spinor_labels())
    # label_ang = np.array([re.search(r'[a-z]', label.split()[2]).group(0) # for label in labels])
    # angular_tag = np.empty(n2c, dtype='U1')
    # for i in range(n2c):
    #     ci = abs(c_tilde[:,i])**2
    #     norm_ang = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    #     angs = np.array(['s','p', 'd', 'f', 'g'])
    #     for j, label in enumerate(angs):
    #         norm_ang[j] = sum(ci[np.where(label_ang == label)])
    #     angular_tag[i] = angs[np.argmax(norm_ang)]

    sort_idx = np.argsort(e)
    irrep_tag = irrep_tag[sort_idx]
    #angular_tag = angular_tag[sort_idx]
    return lib.tag_array(e[sort_idx],irrep_tag=irrep_tag), c[:,sort_idx]
           # lib.tag_array(c[:,sort_idx], ang_tag=angular_tag)

def get_occ_symm(mf, irrep, occup, irrep_mo=None, mo_energy=None, mo_coeff=None):
    mol = mf.mol

    ir_tag = irrep_mo
    if ir_tag is None:
        if mo_energy is None:
            mo_energy = mf.mo_energy
        ir_tag=mf.irrep_mo

    n2c = mo_energy.shape[0]//2
    ir_tag = ir_tag[n2c:] # sort only electronic states
    print(ir_tag)
    mo_occ = np.zeros(n2c, dtype=complex)    
    if mo_coeff is None and mf.mo_coeff is None:
        for ir in irrep:
            ir_mo = np.where(ir_tag==ir)
            if isinstance(occup[ir][0], int): 
                occupy = sum(occup[ir])
            else:
                occupy = occup[ir][0]
            mo_occ[ir_mo[:occupy]] = 1.0
            #mo_occ
            return mo_occ
    elif mo_coeff is None:
        mo_coeff = mf.mo_coeff

    mo_coeff_ll = mo_coeff[:n2c,n2c:]
    angular_momentum = ['s', 'p', 'd', 'f']
    n_ang = 4

    for ir in irrep:
        ir_mo = np.where(ir_tag==ir)[0]
        if not ir in occup:
            continue
        occupy = occup[ir]
        if isinstance(occup[ir][0], int): 
            mo_occ[ir_mo[:occupy[0]]] = 1.0+0.j 
        else:
            mo_occ[ir_mo[occupy[0]]] = 1.0+0.j 
        for i in range(1, min(len(occupy), n_ang+1)):
            if occupy[i] == 0:
                continue
            else:
                ang = angular_momentum[i-1]
                indices = np.where(mo_coeff.ang_tag[ir_mo[occupy[0]:]] == ang)[0]
                if len(indices) < occupy[i]:
                    raise ValueError(f'Not enough mo with largest contribution from {ang} found')
                else:
                    mo_occ[ir_mo[occupy[0]:][indices[:occupy[i]]]] = 1.0+0.j
    occupied = np.where(mo_occ == 1.0+0.j)[0]
    count = [0,0,0,0]
    #for idx, irrep, ene, ang in zip(occupied, ir_tag[occupied], mo_energy[occupied], mo_coeff.ang_tag[occupied]):
    #    if ang == 's':
    #        count[0]+=1
    #for idx, irrep, ene, ang in zip(occupied, ir_tag[occupied], mo_energy[occupied], mo_coeff.ang_tag[occupied]):
    #    if ang == 'p':
    #        count[1]+=1
    #for idx, irrep, ene, ang in zip(occupied, ir_tag[occupied], mo_energy[occupied], mo_coeff.ang_tag[occupied]):
    #    if ang == 'd':
    #        count[2]+=1
    #for idx, irrep, ene, ang in zip(occupied, ir_tag[occupied], mo_energy[occupied], mo_coeff.ang_tag[occupied]):
    #    if ang == 'f':
    #        count[3]+=1
    mo_occ = np.hstack((np.zeros(n2c), mo_occ))
    return mo_occ

# linear symmetry in 4c HF
class SymmDHF(dhf.DHF):
    def __init__(self, mol, symmetry=None, occup=None):
        dhf.DHF.__init__(self, mol)
        if symmetry is 'linear':
            self.irrep_ao = symmetry_label(mol, symmetry)
        self.occupation = occup
        print('occup')
        print(self.occupation)

    def eig(self, h, s):
        e, c = eig(self, h, s, irrep=self.irrep_ao)
        self.irrep_mo = e.irrep_tag
        return e, c
    
    def _eigh(self, h, s):
        return scipy.linalg.eigh(h, s)
    
    def get_occ(self, mo_energy=None, mo_coeff=None):
        if self.occupation is None or mo_energy is None:
            return dhf.DHF.get_occ(self, mo_energy, mo_coeff)
        else:
            return get_occ_symm(self, self.irrep_ao, self.occupation, mo_energy=mo_energy, mo_coeff=mo_coeff)

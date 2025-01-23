import numpy as np
import scipy
from numpy import dot as matmul
from pyscf import gto
from pyscf.lo.iao import reference_mol
from scipy.linalg import sqrtm
from functools import reduce

def sqrtminv(A):
   return scipy.linalg.inv(sqrtm(A))

def SymOrth(C,S=1):
   """symmetrically orthogonalize orbitals C with respect to overlap
   matrix S (i.e., such that Cnew^T S Cnew == id)."""
   return matmul(C, sqrtminv(reduce(matmul, (C.T.conj(),S,C))))

def get_iao_fromS(s1, s2, s12, coeff):
    nbas = s1.shape[0]
    s1inv = scipy.linalg.inv(s1)
    s2inv = scipy.linalg.inv(s2)
    proj12 = matmul(s1inv,s12)
    
    coeff_tilde = SymOrth(reduce(matmul, (s1inv,s12,s2inv,s12.T.conj(),coeff)), s1)
    
    Oi1 = reduce(matmul, (coeff, coeff.T.conj(), s1))
    Oti1 = reduce(matmul, (coeff_tilde, coeff_tilde.T.conj(), s1))
    
    iao = matmul(np.eye(nbas)-Oi1-Oti1+2.0*matmul(Oi1, Oti1), proj12)
    return SymOrth(iao, s1)

def get_iao(mol, mo_occ, spinor = False):
    mol_minao = reference_mol(mol, 'minao')
    if(spinor):
        s1 = mol.intor('int1e_ovlp_spinor')
        s2 = mol_minao.intor('int1e_ovlp_spinor')
        s12 = gto.intor_cross('int1e_ovlp_spinor', mol, mol_minao)
        aoslices = mol_minao.aoslice_2c_by_atom()
    else:
        s1 = mol.intor_symmetric('int1e_ovlp')
        s2 = mol_minao.intor_symmetric('int1e_ovlp')
        s12 = gto.intor_cross('int1e_ovlp', mol, mol_minao)
        aoslices = mol_minao.aoslice_by_atom()
    return get_iao_fromS(s1, s2, s12, mo_occ), aoslices

def get_ibo_scalar(ovlp, ao_slices_minao, coeff, iao, p, maxiter, gradthreshold):
    Ca = reduce(matmul, (iao.T.conj(), ovlp, coeff))
    Ni = Ca.shape[1]
    NAtom = len(ao_slices_minao)
    assert ao_slices_minao[NAtom-1][3] == Ca.shape[0], \
    "# Basis for ao_slices %d != Ca.shape[0] %d" % (ao_slices_minao[NAtom-1][3], Ca.shape[0])
    conv = False
    for iter in range(maxiter):
        grad = 0.0
        for i in range(Ni):
            for j in range(i):
                Avalue = 0.0
                Bvalue = 0.0
                Ci = 1. * Ca[:,i]
                Cj = 1. * Ca[:,j]
                for aa in range(NAtom):
                    astart = ao_slices_minao[aa][2]
                    aend = ao_slices_minao[aa][3]
                    Cia = Ci[astart:aend]
                    Cja = Cj[astart:aend]
                    Qii = matmul(Cia.T.conj(), Cia)
                    Qjj = matmul(Cja.T.conj(), Cja)
                    Qij = matmul(Cia.T.conj(), Cja)
                    Bvalue += 0.5*p*Qij*(Qii**(p-1) - Qjj**(p-1))
                    Avalue += 0.25*p*(p-1)*Qij**2*(Qii**(p-2) + Qjj**(p-2)) \
                            - 0.125*p*(Qii**(p-1) - Qjj**(p-1))*(Qii-Qjj)
                grad += Bvalue**2
                phi = 0.25*np.arctan2(Bvalue,-Avalue)
                cp = np.cos(phi)
                sp = np.sin(phi)
                Ca[:,i] = cp * Ci + sp * Cj
                Ca[:,j] =-sp * Ci + cp * Cj
        grad = grad**0.5
        if(grad < gradthreshold):
            conv = True
            break
    if(conv):
        print("IBO localization converged in %d iterations! Final gradient %.2e" % (iter+1, grad))
    else:
        print("ERROR: IBO localization not converged!")
        exit()
             
    return matmul(iao, SymOrth(Ca))

def get_ibo_spinor(ovlp, ao_slices_minao, coeff, iao, p, maxiter, gradthreshold):
    Ca = reduce(matmul, (iao.T.conj(), ovlp, coeff))
    Ni = Ca.shape[1]
    NAtom = len(ao_slices_minao)
    assert ao_slices_minao[NAtom-1][3] == Ca.shape[0], \
    "# Basis for ao_slices %d != Ca.shape[0] %d" % (ao_slices_minao[NAtom-1][3], Ca.shape[0])
    conv = False
    for iter in range(maxiter):
        grad = 0.0
        for i in range(Ni):
            for j in range(i):
                Ci = 1. * Ca[:,i]
                Cj = 1. * Ca[:,j]

                Avalue = 0.0
                Bvalue = 0.0
                for aa in range(NAtom):
                    astart = ao_slices_minao[aa][2]
                    aend = ao_slices_minao[aa][3]
                    Cia = Ci[astart:aend]
                    Cja = Cj[astart:aend]
                    Qii = (matmul(Cia.T.conj(), Cia)).real
                    Qjj = (matmul(Cja.T.conj(), Cja)).real
                    Qij = matmul(Cia.T.conj(), Cja)
                    ReQij = Qij.real
                    ImQij = Qij.imag
                    
                    Bvalue += (p*Qii**(p-1) - p*Qjj**(p-1))*ImQij
                    Avalue += (p*Qii**(p-1) - p*Qjj**(p-1))*ReQij
                tau = 0.5*np.arctan2(Bvalue,-Avalue)
                c2t = np.cos(2.0*tau)
                s2t = np.sin(2.0*tau)
                
                Avalue = 0.0
                Bvalue = 0.0        
                for aa in range(NAtom):
                    astart = ao_slices_minao[aa][2]
                    aend = ao_slices_minao[aa][3]
                    Cia = Ci[astart:aend]
                    Cja = Cj[astart:aend]
                    Qii = (matmul(Cia.T.conj(), Cia)).real
                    Qjj = (matmul(Cja.T.conj(), Cja)).real
                    Qij = matmul(Cia.T.conj(), Cja)
                    ReExp2itQij = Qij.real*c2t - Qij.imag*s2t
                    Bvalue += 0.5*p*ReExp2itQij*(Qii**(p-1) - Qjj**(p-1))
                    Avalue += 0.25*p*(p-1)*ReExp2itQij**2*(Qii**(p-2) + Qjj**(p-2))\
                            - 0.125*p*(Qii**(p-1) - Qjj**(p-1))*(Qii-Qjj)
                grad += Bvalue**2
                phi = 0.25*np.arctan2(Bvalue,-Avalue)
                cp = np.cos(phi)
                sp = np.sin(phi)
                Ca[:,i] = cp * np.exp(-1.0*complex(0,tau)) * Ci + sp * np.exp(1.0*complex(0,tau)) * Cj
                Ca[:,j] =-sp * np.exp(-1.0*complex(0,tau)) * Ci + cp * np.exp(1.0*complex(0,tau)) * Cj
        grad = grad**0.5
        print("IBO localization iteration %d, gradient %.2e" % (iter+1, grad))
        if(grad < gradthreshold):
            conv = True
            break
    if(conv):
        print("IBO localization converged in %d iterations! Final gradient %.2e" % (iter+1, grad))
    else:
        print("ERROR: IBO localization not converged!")
        exit()
             
    return matmul(iao, SymOrth(Ca))

def get_ibo(mol, coeff_occ, p=4, maxiter = 200, gradthreshold = 1e-8, spinor = False):
    mol_minao = reference_mol(mol, 'minao')
    if(spinor):
        s1 = mol.intor('int1e_ovlp_spinor')
        s2 = mol_minao.intor('int1e_ovlp_spinor')
        s12 = gto.intor_cross('int1e_ovlp_spinor', mol, mol_minao)
        ao_slices_minao = mol_minao.aoslice_2c_by_atom()
    else:
        s1 = mol.intor('int1e_ovlp')
        s2 = mol_minao.intor('int1e_ovlp')
        s12 = gto.intor_cross('int1e_ovlp', mol, mol_minao)
        ao_slices_minao = mol_minao.aoslice_by_atom()
    iao = get_iao_fromS(s1, s2, s12, coeff_occ)
    
    if(spinor):
        return get_ibo_spinor(s1, ao_slices_minao, coeff_occ, iao, p, maxiter, gradthreshold)
    else:
        return get_ibo_scalar(s1, ao_slices_minao, coeff_occ, iao, p, maxiter, gradthreshold)
    

if __name__ == "__main__":
    from socutils.scf import spinor_hf
    from socutils.somf import amf
    benzene = [[ 'C'  , ( 4.673795 ,   6.280948 , 0.00  ) ],
           [ 'C'  , ( 5.901190 ,   5.572311 , 0.00  ) ],
           [ 'C'  , ( 5.901190 ,   4.155037 , 0.00  ) ],
           [ 'C'  , ( 4.673795 ,   3.446400 , 0.00  ) ],
           [ 'C'  , ( 3.446400 ,   4.155037 , 0.00  ) ],
           [ 'C'  , ( 3.446400 ,   5.572311 , 0.00  ) ],
           [ 'H'  , ( 4.673795 ,   7.376888 , 0.00  ) ],
           [ 'H'  , ( 6.850301 ,   6.120281 , 0.00  ) ],
           [ 'H'  , ( 6.850301 ,   3.607068 , 0.00  ) ],
           [ 'H'  , ( 4.673795 ,   2.350461 , 0.00  ) ],
           [ 'H'  , ( 2.497289 ,   3.607068 , 0.00  ) ],
           [ 'H'  , ( 2.497289 ,   6.120281 , 0.00  ) ]]

    mol = gto.M(atom=benzene,
                basis='ccpvdz')
    mf = spinor_hf.JHF(mol)
    mf.with_x2c = amf.SpinorX2CAMFHelper(mol, with_gaunt=True, with_breit=True)
    mf.kernel()
    mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
    ibos = get_ibo(mol, mo_occ, spinor=True)


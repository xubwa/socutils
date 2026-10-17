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

def get_iao_fromS(s1, s2, s12, coeff, orthogonalize=True):
    nbas = s1.shape[0]
    s1inv = scipy.linalg.inv(s1)
    s2inv = scipy.linalg.inv(s2)
    proj12 = matmul(s1inv,s12)

    coeff_tilde = SymOrth(reduce(matmul, (s1inv,s12,s2inv,s12.T.conj(),coeff)), s1)

    Oi1 = reduce(matmul, (coeff, coeff.T.conj(), s1))
    Oti1 = reduce(matmul, (coeff_tilde, coeff_tilde.T.conj(), s1))

    iao = matmul(np.eye(nbas)-Oi1-Oti1+2.0*matmul(Oi1, Oti1), proj12)
    if not orthogonalize:
        # non-orthogonal IAOs keep a 1:1 per-reference-AO character, needed
        # when individual columns are selected afterwards (e.g. AVAS)
        return iao
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

def time_reversal(mol, coeff):
    '''Apply the time-reversal operator T to spinor orbital coefficients.'''
    trmap = mol.time_reversal_map()
    idx = np.abs(trmap) - 1
    sgn = np.sign(trmap)
    tc = np.zeros_like(coeff)
    tc[idx] = sgn[:, None] * coeff.conj()
    return tc

def kramers_pair(mol, orbs, ovlp=None, tol=1e-6):
    '''Rotate an orthonormal set of spinor orbitals spanning a time-reversal
    closed space into explicit Kramers pairs, returned as adjacent columns
    (psi_1, T psi_1, psi_2, T psi_2, ...).

    Localized orbitals from a Kramers-restricted reference are degenerate in
    the localization objective within each set of orbitals sharing the same
    atomic populations, so the converged orbitals are generally *not* Kramers
    partners of each other.  This post-processing only mixes orbitals within
    such degenerate blocks (atomic populations are invariant), pairing the
    "purest" orbitals first: each iteration picks the orbital whose T image
    is closest to a single remaining partner, keeps it and its exact T image,
    and projects both out of the rest.
    '''
    if ovlp is None:
        ovlp = mol.intor('int1e_ovlp_spinor')
    nmo = orbs.shape[1]
    assert nmo % 2 == 0, 'Kramers pairing needs an even number of orbitals'

    # the span must be closed under T (Kramers-restricted reference)
    m = reduce(matmul, (orbs.T.conj(), ovlp, time_reversal(mol, orbs)))
    closure = np.linalg.norm(m, axis=0)
    if abs(closure - 1.0).max() > tol:
        raise ValueError('orbital span is not closed under time reversal '
                         '(max deviation %.2e); a Kramers-restricted set of '
                         'orbitals is required' % abs(closure - 1.0).max())

    rem = orbs.copy()
    paired = []
    while rem.shape[1] > 0:
        trem = time_reversal(mol, rem)
        m = reduce(matmul, (rem.T.conj(), ovlp, trem))
        i = np.argmax(np.abs(m).max(axis=0))
        v = rem[:, i]
        w = trem[:, i]
        paired += [v, w]
        if rem.shape[1] == 2:
            break
        # v is a member of rem, so only w overlaps the remaining orbitals.
        # Drop the remaining orbital most similar to w, project w out of the
        # others, and re-orthonormalize with the minimal-perturbation Loewdin
        # scheme.  (Canonical orthonormalization is unusable here: the
        # remaining overlap matrix is near-identity, its eigenvectors within
        # the degenerate eigenvalue 1 are arbitrary and would mix orbitals
        # across different localization blocks.)
        rest = np.delete(rem, i, axis=1)
        wov = matmul(w.conj(), matmul(ovlp, rest))
        rest = np.delete(rest, np.argmax(np.abs(wov)), axis=1)
        rest = rest - np.outer(w, matmul(w.conj(), matmul(ovlp, rest)))
        rem = SymOrth(rest, ovlp)
    return np.array(paired).T

def get_ibo(mol, coeff_occ, p=4, maxiter = 200, gradthreshold = 1e-8, spinor = False, kramers = True):
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
        ibos = get_ibo_spinor(s1, ao_slices_minao, coeff_occ, iao, p, maxiter, gradthreshold)
        if kramers:
            ibos = kramers_pair(mol, ibos, ovlp=s1)
        return ibos
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


import numpy as np
from functools import reduce
from pyscf import scf
from pyscf.data.nist import LIGHT_SPEED
from socutils.scf import spinor_hf

def mmf4c(dhf, contr_coeff = None):
    assert(isinstance(dhf, scf.dhf.DHF))
    if contr_coeff is not None:
        raise NotImplementedError("contraction projection is not implemented yet.")
    return mmf4c_fock(dhf.mol, dhf.mo_coeff, dhf.get_fock(), dhf.get_ovlp())

def mmf4c_fock(mol, coeff4c, fockao_4c, ovlp_4c = None):
    assert(coeff4c.shape == fockao_4c.shape)
    assert(coeff4c.shape[0] == mol.nao_2c() * 2)

    n2c = mol.nao_2c()
    ene_4c = reduce(np.dot, (coeff4c.T.conj(), fockao_4c, coeff4c)).diagonal().real

    if ovlp_4c is None:
        s_ll = mol.intor('int1e_ovlp_spinor')
        s_ss = mol.intor('int1e_kin_spinor') / 2.0 / LIGHT_SPEED**2
    else:
        s_ll = ovlp_4c[:n2c,:n2c]
        s_ss = ovlp_4c[n2c:,n2c:]
    from socutils.somf.x2c_grad import x2c1e_hfw0_block
    _, _, _, _, r, _, _, _ = x2c1e_hfw0_block(fockao_4c[:n2c,:n2c], fockao_4c[n2c:,:n2c], fockao_4c[:n2c,n2c:],
                                                    fockao_4c[n2c:,n2c:], s_ll, s_ss)
    rinv = np.linalg.inv(r)
    
    jhf = spinor_hf.JHF(mol)
    jhf.converged = True
    jhf.mo_occ = np.zeros(n2c)
    jhf.mo_occ[:mol.nelectron] = 1
    jhf.mo_energy = ene_4c[n2c:]
    jhf.mo_coeff = np.dot(rinv, coeff4c[:n2c, n2c:])

    return jhf

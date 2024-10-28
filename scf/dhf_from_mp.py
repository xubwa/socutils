# this file creates a shortcut to do a DHF calculation using a converged
# model potential calculation as the initial guess.
import numpy as np
from socutils.somf import eamf
from socutils.scf import spinor_hf

def dhf_from_mp(dhf):
    mol = dhf.mol
    gaunt = dhf.with_gaunt
    breit = dhf.with_breit
    mp = spinor_hf.SpinorSCF(mol)
    mp.with_x2c = eamf.SpinorEAMFX2CHelper(mol, eamf='eamf', with_gaunt=gaunt, with_breit=breit)
    mp.kernel()
    c4c = mp.with_x2c.to_4c_coeff(mp.mo_coeff)
    n2c = mol.nao_2c()
    mo_occ = np.hstack((np.zeros(n2c), mp.get_occ()))
    dm4c = c4c@np.diag(mo_occ)@c4c.T.conj()
    dhf.kernel(dm0=dm4c)
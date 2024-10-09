import re
import numpy as np
from socutils.tools.spinor2sph import spinor2sph
from socutils.scf.spinor_hf import sph2spinor
# assumes only primitives
def add_diffuse(bas, angular=[0,1,2], expand=1, multiplier=2.5):
    for ang in angular:
        prim_list = []
        for prim in bas:
            if prim[0] == ang:
                prim_list.append(prim[-1][0])
        prim_min = min(prim_list)
        for iexpand in range(expand):
            bas.append([ang,[prim_min*(1./multiplier)**(iexpand+1),1.0]])
    return bas
def construct_configuration(sph_irrep,configuration='2s1s'):
    angulars = ['s','p','d','f','g','h','i']
    ang_shift = [1,2,3,4,5,6,7]
    shell_size = [2,6,10,14,18,22,26]
    reorder = []
    offset = []
    for ang in angulars:
        offset.append(len(reorder))
        for i, sph in enumerate(sph_irrep):
            if ang in sph:
                reorder.append(i)
    config = re.findall(r'\d+[a-z]', configuration)
    result = []
    print(reorder)
    for iconfig in config:
        n = int(iconfig[0:-1])
        ang = angulars.index(iconfig[-1])
        for i in range(shell_size[ang]):
            result.append(reorder[offset[ang]+(n-ang_shift[ang])*shell_size[ang]+i])
    for idx in reorder:
        if idx not in result:
            result.append(idx)
    print(result)
    return result

def ghf2jhf_dm(mol, dm_ghf):
    r"""
    Convert GHF density matrix to JHF density matrix
    """
    return sph2spinor(mol, dm_ghf)
def ghf2jhf_chkfile(mol, chkfile_name):
    r"""
    Convert GHF chkfile to JHF density matrix
    """
    from pyscf.scf.ghf import init_guess_by_chkfile
    dm_ghf = init_guess_by_chkfile(mol, chkfile_name)
    return ghf2jhf_dm(mol, dm_ghf)

def jhf2ghf_dm(mol, dm_jhf):
    r"""
    Convert JHF density matrix to GHF density matrix
    """
    c = np.vstack(mol.sph2spinor_coeff())
    return np.dot(c, np.dot(dm_jhf, c.T.conj()))
def jhf2ghf_chkfile(mol, chkfile_name):
    r"""
    Convert JHF chkfile to GHF density matrix
    """
    from socutils.scf.spinor_hf import init_guess_by_chkfile
    dm_jhf = init_guess_by_chkfile(mol, chkfile_name)
    return jhf2ghf_dm(mol, dm_jhf)

def jhf2dhf_dm(mol, dm_ghf):
    r"""
    Convert GHF density matrix to DHF density matrix
    """
    n2c = dm_ghf.shape[0]
    dm_dhf = np.zeros((n2c*2,n2c*2),dtype=dm_ghf.dtype)
    dm_dhf[:n2c,:n2c] = dm_ghf
    return dm_dhf
def ghf2dhf_chkfile(mol, chkfile_name):
    r"""
    Convert GHF chkfile to DHF density matrix
    """
    dm_jhf = ghf2jhf_chkfile(mol, chkfile_name)
    return jhf2dhf_dm(mol, dm_jhf)
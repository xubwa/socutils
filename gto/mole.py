# override mole object in pyscf to handle spin-orbit contraction
# Xubo Wang 2024/3/1
import numpy as np

from pyscf import gto
from pyscf import __config__

ARGPARSE = getattr(__config__, 'ARGPARSE', False)

def condense_shell(info):
    # information for contracted basis
    # 1st element is angular momentum
    # 2nd element is number of j=l-1/2 basis
    # 3rd element is number of j=l+1/2 basis
    # 4th element is number of shared basis
    ang_mom = info[0] * 2 + 1
    nbas_kminus = info[0] * 2
    nbas_kplus = info[0] * 2 + 2
    nbas_shared = ang_mom * 2
    ngto = info[1] + info[2] + info[3]
    nbas_real = ngto * ang_mom * 2
    nbas_spinor = info[1]*nbas_kminus + info[2]*nbas_kplus + info[3]*ang_mom*2
    print(nbas_real, nbas_spinor)
    out = np.zeros((nbas_real, nbas_spinor))
    
    block_kminus = np.zeros((nbas_shared, nbas_kminus))
    block_kplus = np.zeros((nbas_shared, nbas_kplus))
    block_shared = np.zeros((nbas_shared, nbas_shared))
    for i in range(nbas_kminus):
        block_kminus[i,i] = 1
    for i in range(nbas_kplus):
        block_kplus[i+nbas_kminus,i] = 1
    for i in range(ang_mom*2):
        block_shared[i,i] = 1

    for i in range(info[1]):
        out[i*ang_mom*2:(i+1)*ang_mom*2, i*nbas_kminus:(i+1)*nbas_kminus] = block_kminus
    offset_axis1 = info[1]*nbas_shared
    offset_axis2 = info[1]*nbas_kminus
    for i in range(info[2]):
        axis1_start = offset_axis1+i*nbas_shared
        axis2_start = offset_axis2+i*nbas_kplus
        out[axis1_start:axis1_start+nbas_shared, axis2_start:axis2_start+nbas_kplus] = block_kplus
    offset_axis1 += info[2]*nbas_shared
    offset_axis2 += info[2]*nbas_kplus
    for i in range(info[3]):
        axis1_start = offset_axis1+i*nbas_shared
        axis2_start = offset_axis2+i*nbas_shared
        out[axis1_start:axis1_start+nbas_shared, axis2_start:axis2_start+nbas_shared] = block_shared
    return out

def condense_atom(ctr_scheme):
    condense = []
    for ctr_shell in ctr_scheme:
        condense.append(condense_shell(ctr_shell))
    nscalar = sum(condense_i.shape[0] for condense_i in condense)
    nspinor = sum(condense_i.shape[1] for condense_i in condense)
    out = np.zeros((nscalar, nspinor))
    offset_scalar = 0
    offset_spinor = 0
    for condense_i in condense:
        nscalar_i, nspinor_i = condense_i.shape
        out[offset_scalar:offset_scalar+nscalar_i, offset_spinor:offset_spinor+nspinor_i] = condense_i
        offset_scalar += nscalar_i
        offset_spinor += nspinor_i
    return out

def condense_mol(mol, mol_scheme):
    bas_info = mol._bas
    nscalar_2c = mol.nao_2c()
    ao_spinor_list = []
    ao_slice_2c = mol.aoslice_2c_by_atom()
    spinor_slice_2c = np.zeros_like(ao_slice_2c)
    for atom_id, atom_scheme in enumerate(mol_scheme):
        if atom_scheme == 0:
            ao_spinor_list.append(np.eye(ao_slice_2c[atom_id,3]-ao_slice_2c[atom_id,2]))
        else: 
            ao_spinor_list.append(condense_atom(atom_scheme))

        if atom_id == 0:
            spinor_slice_2c[atom_id,2] = 0
        else:
            spinor_slice_2c[atom_id,2] = spinor_slice_2c[atom_id-1,3]

        spinor_slice_2c[atom_id,3] = spinor_slice_2c[atom_id,2]+ao_spinor_list[-1].shape[1]

    out = np.zeros((nscalar_2c, spinor_slice_2c[-1,3]))
    for atom_id, atom_contr in enumerate(ao_spinor_list):
        out[ao_slice_2c[atom_id,2]:ao_slice_2c[atom_id,3], spinor_slice_2c[atom_id,2]:spinor_slice_2c[atom_id,3]] = atom_contr
    return out

class Mole(gto.Mole):
    def __init__(self):
        super().__init__()
        self.sobasis_info = None
        self.so_contr = None
    def build(self, dump_input=True, parse_arg=ARGPARSE,
              verbose=None, output=None, max_memory=None,
              atom=None, basis=None, unit=None, nucmod=None, ecp=None, pseudo=None,
              charge=None, spin=0, symmetry=None, symmetry_subgroup=None,
              cart=None, magmom=None, sobasis_info=None):
        gto.Mole.build(self, dump_input=dump_input, parse_arg=parse_arg,
              verbose=verbose, output=output, max_memory=max_memory,
              atom=atom, basis=basis, unit=unit, nucmod=nucmod, ecp=ecp, pseudo=pseudo,
              charge=charge, spin=spin, symmetry=symmetry, symmetry_subgroup=symmetry_subgroup,
              cart=cart, magmom=magmom)
        if sobasis_info is not None:
            self.so_contr = condense_mol(self, sobasis_info)

    def sph2spinor_coeff(self):
        from pyscf.symm import sph
        c1, c2 = sph.sph2spinor_coeff(self)
        c1 = np.dot(c1, self.so_contr)
        c2 = np.dot(c2, self.so_contr)
        return c1, c2

            

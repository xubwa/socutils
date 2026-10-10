"""Writes a many-electron test Hamiltonian: Cl2, spinor Hartree-Fock of
socutils in the 6-31G basis, all 34 electrons in all 52 spinors.

Needs pyscf and socutils. Run:  python make_fcidump.py
"""
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.tools import fcidump_rel

mol = gto.M(atom='Cl 0 0 0; Cl 0 0 1.988', basis='6-31g', verbose=3)
mf = spinor_hf.SpinorSCF(mol)
mf.kernel()
assert mf.converged
num_spinors = mol.nao_2c()
fcidump_rel.from_x2c(mf, 0, num_spinors, filename='FCIDUMP', tol=1e-12)
print('spinors', num_spinors, 'electrons', mol.nelectron, 'E(HF)', mf.e_tot)

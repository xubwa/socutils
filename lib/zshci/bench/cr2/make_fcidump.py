"""Writes a large relativistic test Hamiltonian: Cr2 at 1.68 Angstrom, spinor
Hartree-Fock of socutils in the def2-SVP basis, the 24 electrons above the
1s2s2p3s shells of both atoms in the 60 spinors that follow the 24 frozen
ones (the spinor counterpart of the (24e, 30o) active space often used for
this molecule).

Needs pyscf and socutils. Run:  python make_fcidump.py
"""
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.tools import fcidump_rel

NUM_CORE = 24
NUM_ACTIVE = 60

mol = gto.M(atom='Cr 0 0 0; Cr 0 0 1.68', basis='def2-svp', verbose=4)
mf = spinor_hf.SpinorSCF(mol)
mf.level_shift = 0.3
mf.max_cycle = 200
mf.kernel()
assert mf.converged
fcidump_rel.from_x2c(mf, NUM_CORE, NUM_ACTIVE, filename='FCIDUMP', tol=1e-12)
print('spinors', mol.nao_2c(), 'active', NUM_ACTIVE, 'electrons',
      mol.nelectron - NUM_CORE, 'E(HF)', mf.e_tot)

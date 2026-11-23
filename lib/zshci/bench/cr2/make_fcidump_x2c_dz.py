"""Writes the Hamiltonian of the largest published SHCI benchmark (Li, Otten,
Holmes, Sharma and Umrigar, J. Chem. Phys. 149, 214110 (2018), table III):
Cr2 at 1.68 Angstrom, cc-pVDZ-DK basis, scalar X2C Hamiltonian, the 1s2s2p
shells frozen and the 28 other electrons in all 76 remaining orbitals, as a
standard FCIDUMP file over spatial orbitals with D2h labels ("orbitals":
"spatial" of zshci).

    python make_fcidump_x2c_dz.py [file]

The paper uses orbitals that minimize the variational energy at
epsilon_1 = 2e-4; these are the Hartree-Fock orbitals, so the energies and
the numbers of determinants at a given epsilon_1 differ from the table.

Needs pyscf.
"""
import sys

from pyscf import gto, mcscf, scf
from pyscf.tools import fcidump

filename = sys.argv[1] if len(sys.argv) > 1 else 'FCIDUMP_x2c_dz_28e_76'

mol = gto.M(atom='Cr 0 0 0; Cr 0 0 1.68', basis='cc-pvdz-dk', symmetry='D2h',
            verbose=4)
mf = scf.RHF(mol).sfx2c1e()
mf.level_shift = 0.3
mf.max_cycle = 200
mf.kernel()
assert mf.converged
num_core = 10
num_active = mol.nao_nr() - num_core
num_elec = mol.nelectron - 2 * num_core
mc = mcscf.CASCI(mf, num_active, num_elec)
fcidump.from_mcscf(mc, filename, tol=1e-12, molpro_orbsym=True)
print('orbitals', mol.nao_nr(), 'active', num_active, 'electrons', num_elec,
      'E(HF)', mf.e_tot)

"""Writes the non-relativistic counterpart of make_fcidump.py: Cr2 at 1.68
Angstrom, restricted Hartree-Fock in the def2-SVP basis with D2h symmetry,
the 24 electrons above the 1s2s2p3s shells in the next 30 orbitals, as a
standard FCIDUMP file over spatial orbitals with the irreducible
representations in Molpro's numbering ("orbitals": "spatial" of zshci).

    python make_fcidump_rhf.py [num_core_orbitals [num_active_orbitals [file]]]

Needs pyscf.
"""
import sys

from pyscf import gto, mcscf, scf
from pyscf.tools import fcidump

num_core = int(sys.argv[1]) if len(sys.argv) > 1 else 12
num_active = int(sys.argv[2]) if len(sys.argv) > 2 else 30
filename = sys.argv[3] if len(sys.argv) > 3 else 'FCIDUMP_rhf'

mol = gto.M(atom='Cr 0 0 0; Cr 0 0 1.68', basis='def2-svp', symmetry='D2h',
            verbose=4)
mf = scf.RHF(mol)
mf.level_shift = 0.3
mf.max_cycle = 200
mf.kernel()
assert mf.converged
num_elec = mol.nelectron - 2 * num_core
mc = mcscf.CASCI(mf, num_active, num_elec)
fcidump.from_mcscf(mc, filename, tol=1e-12, molpro_orbsym=True)
print('orbitals', mol.nao_nr(), 'active', num_active, 'electrons', num_elec,
      'E(HF)', mf.e_tot)

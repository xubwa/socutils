"""Writes large relativistic test Hamiltonians: Cr2 at 1.68 Angstrom, spinor
Hartree-Fock of socutils in the def2-SVP basis (124 spinors, 48 electrons).

    python make_fcidump.py [num_core [num_active [file]]]

freezes the lowest num_core spinors and takes the next num_active ones. The
default, 24 and 60, is the 24 electrons above the 1s2s2p3s shells of both
atoms in 60 spinors, the spinor counterpart of the (24e, 30o) active space
often used for this molecule. More virtual spinors: 24 100 (all of them).
More occupied ones: 12 72 unfreezes 2p and 3s (36 electrons).

Needs pyscf and socutils.
"""
import sys

from pyscf import gto
from socutils.scf import spinor_hf
from socutils.tools import fcidump_rel

num_core = int(sys.argv[1]) if len(sys.argv) > 1 else 24
num_active = int(sys.argv[2]) if len(sys.argv) > 2 else 60
filename = sys.argv[3] if len(sys.argv) > 3 else 'FCIDUMP'

mol = gto.M(atom='Cr 0 0 0; Cr 0 0 1.68', basis='def2-svp', verbose=4)
mf = spinor_hf.SpinorSCF(mol)
mf.level_shift = 0.3
mf.max_cycle = 200
mf.kernel()
assert mf.converged
assert num_core + num_active <= mol.nao_2c()
fcidump_rel.from_x2c(mf, num_core, num_active, filename=filename, tol=1e-12)
print('spinors', mol.nao_2c(), 'active', num_active, 'electrons',
      mol.nelectron - num_core, 'E(HF)', mf.e_tot)

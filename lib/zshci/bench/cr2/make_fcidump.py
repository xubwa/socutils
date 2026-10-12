"""Writes large relativistic test Hamiltonians: Cr2 at 1.68 Angstrom in the
def2-SVP basis (124 spinors, 48 electrons), spinor Hartree-Fock of socutils
with the X2CAMF Hamiltonian (Gaunt and Breit terms in the mean field) and
the symmetry of a linear molecule, so that every spinor has a definite m_j
about the axis. ORBSYM holds 2 m_j, which zshci conserves.

    python make_fcidump.py [num_core [num_active [file [options]]]]

freezes the lowest num_core spinors and takes the next num_active ones. The
default, 24 and 60, is the 24 electrons above the 1s2s2p3s shells of both
atoms in 60 spinors, the spinor counterpart of the (24e, 30o) active space
often used for this molecule. More virtual spinors: 24 100 (all of them).
More occupied ones: 12 72 unfreezes 2p and 3s (36 electrons).

Options, a comma-separated list: "nonrel" leaves the relativistic terms out
(the non-relativistic Hamiltonian in spinors), "nosym" runs the
Hartree-Fock without symmetry, which leaves degenerate spinors mixed and
gives no labels. Both make the Hamiltonian harder for the program and are
meant for stress tests.

Needs pyscf and socutils.
"""
import sys
from fractions import Fraction

from pyscf import gto
from socutils.scf import spinor_hf
from socutils.tools import fcidump_rel

num_core = int(sys.argv[1]) if len(sys.argv) > 1 else 24
num_active = int(sys.argv[2]) if len(sys.argv) > 2 else 60
filename = sys.argv[3] if len(sys.argv) > 3 else 'FCIDUMP'
options = sys.argv[4].split(',') if len(sys.argv) > 4 else []

mol = gto.M(atom='Cr 0 0 0; Cr 0 0 1.68', basis='def2-svp', verbose=4)
if 'nosym' in options:
    mf = spinor_hf.SpinorSCF(mol)
else:
    mf = spinor_hf.SymmSpinorSCF(mol, symmetry='linear')
if 'nonrel' not in options:
    mf = mf.x2camf()
mf.level_shift = 0.3
mf.max_cycle = 200
mf.kernel()
assert mf.converged
assert num_core + num_active <= mol.nao_2c()
labels = None
if 'nosym' not in options:
    labels = [int(2 * Fraction(str(tag))) for tag in mf.irrep_mo]
    assert sum(labels[:num_core]) == 0
    labels = labels[num_core:num_core + num_active]
fcidump_rel.from_x2c(mf, num_core, num_active, filename=filename, tol=1e-12,
                     symm=labels)
print('spinors', mol.nao_2c(), 'active', num_active, 'electrons',
      mol.nelectron - num_core, 'E(HF)', mf.e_tot)
print('2 m_j of the active spinors', labels)

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

Options, a comma-separated list, to make the Hamiltonian harder for the
program:
  c=X     multiplies the speed of light by X. Spin-orbit coupling grows as
          1/X^2: with X = 0.3 chromium has about the coupling of a 6p
          element, and determinants that differ in the spin orientation of
          an electron mix strongly.
  he      adds a helium atom off the axis and off the middle, which leaves
          a mirror plane as the only symmetry; implies nosym.
  nosym   runs the Hartree-Fock without symmetry: no labels, and for the
          linear molecule degenerate spinors come out mixed.
  nonrel  leaves the relativistic terms out (the non-relativistic
          Hamiltonian in spinors).
  shift=X level shift of the Hartree-Fock iterations (default 0.3).
  loose   accepts a Hartree-Fock that has not converged.

Needs pyscf and socutils.
"""
import sys
from fractions import Fraction

from pyscf import gto, lib
from socutils.scf import spinor_hf
from socutils.tools import fcidump_rel

num_core = int(sys.argv[1]) if len(sys.argv) > 1 else 24
num_active = int(sys.argv[2]) if len(sys.argv) > 2 else 60
filename = sys.argv[3] if len(sys.argv) > 3 else 'FCIDUMP'
options = sys.argv[4].split(',') if len(sys.argv) > 4 else []

atoms = 'Cr 0 0 0; Cr 0 0 1.68'
if 'he' in options:
    atoms += '; He 1.8 0 0.3'
    options.append('nosym')
for option in options:
    if option.startswith('c='):
        lib.param.LIGHT_SPEED *= float(option[2:])
mol = gto.M(atom=atoms, basis='def2-svp', verbose=4)
if 'nosym' in options:
    mf = spinor_hf.SpinorSCF(mol)
else:
    mf = spinor_hf.SymmSpinorSCF(mol, symmetry='linear')
if 'nonrel' not in options:
    mf = mf.x2camf()
mf.level_shift = 0.3
for option in options:
    if option.startswith('shift='):
        mf.level_shift = float(option[6:])
mf.max_cycle = 400
mf.kernel()
# Orbitals of an unconverged Hartree-Fock still span a valid test
# Hamiltonian; they are just not the Hartree-Fock orbitals.
assert mf.converged or 'loose' in options
assert num_core + num_active <= mol.nao_2c()
labels = None
if 'nosym' not in options:
    labels = [int(2 * Fraction(str(tag))) for tag in mf.irrep_mo]
    assert sum(labels[:num_core]) == 0
    labels = labels[num_core:num_core + num_active]
fcidump_rel.from_x2c(mf, num_core, num_active, filename=filename, tol=1e-12,
                     symm=labels)
print('spinors', mol.nao_2c(), 'active', num_active, 'electrons',
      mol.nelectron - num_core, 'E(HF)', mf.e_tot, 'speed of light',
      lib.param.LIGHT_SPEED)
print('2 m_j of the active spinors', labels)

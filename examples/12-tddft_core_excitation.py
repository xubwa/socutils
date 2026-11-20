#!/usr/bin/env python
'''
Core-excitation (X-ray absorption, XAS) spectra with spinor TDDFT.

A 2-component (j-adapted spinor) Kohn-Sham reference carries spin-orbit
coupling variationally, so its TDA response gives the relativistic L-edge fine
structure -- the 2p_{1/2} (L2) / 2p_{3/2} (L3) split -- directly, with no
perturbative SOC on top.

Two ingredients:

  * socutils.dft.dft.SpinorDFT(mol, xc=...).x2camf()  -- the spinor KS reference
    (the DFT analogue of spinor_hf.SCF(mol).x2camf(); see example 00).

  * mf.TDA().cvs(core)  -- core-valence separation.  Ordinary TDA returns the
    lowest (valence) excitations; .cvs() freezes every occupied orbital except
    the chosen deep-core spinors, so the solver targets the core edge instead.
    `core` is the list of occupied spinor indices that act as holes.

Deep-core spinors come in j-shells: 2p_{1/2} is a Kramers pair (2 spinors),
2p_{3/2} is a quartet (4 spinors).  In the canonical energy ordering a p-block
core is therefore  ... 2s,2s, 2p1/2,2p1/2, 2p3/2,2p3/2,2p3/2,2p3/2 -- so the
L3 hole space is the four spinors just above the L2 pair (printed below).

Note on the response kernel:
  The response kernel is non-collinear by default (td.collinear = 'auto':
  'ncol' for this closed-shell reference -- m = 0 by Kramers symmetry, where it
  equals the multi-collinear kernel -- and 'mcol' for open shells), so the
  spin-flip channel that the L2/L3 fine structure needs is included even though
  the ground state uses the collinear functional.  td.collinear = 'col' gives
  the old diagonal, spin-conserving kernel for comparison.  An ALDA response
  kernel on top of a hybrid ground state is td.xc_kernel = 'LDA,VWN'.

ccpvdz here is only for a fast, self-contained demo; a core-excitation needs a
core-decontracted / core-polarized basis on the absorbing atom for real numbers.
'''
from pyscf import gto
from socutils.dft import dft as spinor_dft

au2ev = 27.211386245988

mol = gto.M(atom='H 0 0 0; Br 0 0 1.414', basis='ccpvdz', verbose=4)

# spinor X2CAMF Kohn-Sham reference (PBE0); density fitting for the ERIs
mf = spinor_dft.SpinorDFT(mol, xc='pbe0').x2camf().density_fit()
mf.kernel()
print('E(spinor PBE0) = %.10f  converged = %s' % (mf.e_tot, mf.converged))

# Identify the Br 2p core from the occupied orbital energies.  The deep cores
# come out as  1s,1s | 2s,2s | 2p1/2,2p1/2 | 2p3/2 x4 :
print('\nlowest 10 occupied spinor energies (Hartree):')
for i in range(10):
    print('  %2d  %14.5f' % (i, mf.mo_energy[i]))
l2_core = [4, 5]        # Br 2p_{1/2} pair   -> L2 edge
l3_core = [6, 7, 8, 9]  # Br 2p_{3/2} quartet -> L3 edge


def run_edge(name, core, nstates=6):
    td = mf.TDA().cvs(core)        # freeze all occ except `core` (the holes)
    td.nstates = nstates
    td.kernel()
    f = td.oscillator_strength()   # spinor transition dipole, |mu|^2 = mu.mu*
    print('\n=== %s edge: holes in spinors %s ===' % (name, core))
    print('  %-6s %14s %14s' % ('root', 'energy/eV', 'osc.str.'))
    for k, (e, fk) in enumerate(zip(td.e, f)):
        print('  %-6d %14.3f %14.4e' % (k, e * au2ev, fk))
    return td


run_edge('L3 (2p3/2)', l3_core)
run_edge('L2 (2p1/2)', l2_core)

# The L2/L3 separation in the stick spectra reflects the 2p spin-orbit
# splitting carried by the spinor reference -- the whole point of doing the
# core-excitation two-component rather than scalar + perturbative SOC.

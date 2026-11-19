#!/usr/bin/env python
'''
An open-shell ion in a cell: fractional-occupation SCF, CAHF, CASCI.

The route used for lanthanide dopants (4f^n), shown here for an oxygen atom
(2p^4 over the six 2p spinors) in a box:

  1. Kramers-restricted SCF with the open shell fractionally occupied.  The
     open-shell spinors are chosen by their character (projection on the AOs
     of that shell), not by their energy order.
  2. CAHF: orbital optimization for the average of all determinants of the
     open-shell configuration, with the true Hamiltonian (cahf_fock: only J/K
     of the core and open-shell densities, no transformed integrals).
  3. CASCI in the open shell: the J levels and their spin-orbit splittings.
'''
import numpy
from pyscf import gto, df
from socutils.scf import spinor_hf, addons
from socutils.mcscf import cahf_fock, zcasci
from socutils.pbc import spinor_scf, coulomb_reference
from socutils.pbc.rsdf import RSDF

mol = gto.M(atom='O 0 0 0', basis='cc-pvdz', spin=2, verbose=0)
auxbasis = df.addons.aug_etb(mol, beta=2.0)
cell = coulomb_reference.molecule_in_box(mol, 20.)             # a cubic pyscf Cell, edge 20 bohr
cell.verbose = 4

rsdf = RSDF(cell, auxbasis, precision=1e-11, nuc_zeta=1e14).build()

norb, nelec = 6, 4                                             # the 2p spinors, 2p^4

# 1. fractional-occupation Kramers-restricted HF
mf = spinor_scf.periodic(spinor_hf.KRHF(cell), rsdf).x2camf()
open_aos = addons.spinor_ao_indices(cell, 0, l=1)              # spinor AOs of the p shells of atom 0
addons.projected_frac_occ(mf, open_aos, norb, nelec)
mf.kernel()
w = addons.subspace_weights(mf.get_ovlp(), mf.mo_coeff, open_aos)
print('2p character of the open-shell spinors: %.3f .. %.3f'
      % (w[mf.open_shell_mo].min(), w[mf.open_shell_mo].max()))

# 2. CAHF from the fractional-occupation orbitals (core first, then the open shell)
mo, ncore = addons.cas_order(mf)
mc = cahf_fock.CAHF(mf, norb, nelec, ncore=ncore)
mc.kernel(mo)
print('E(CAHF) = %.10f' % mc.e_tot)

# 3. CASCI(4e, 6 spinors) on the CAHF orbitals: 3P_2, 3P_1, 3P_0, 1D_2, 1S_0 (15 states)
cas = zcasci.CASCI(mf, norb, nelec, ncore=ncore)
cas.fcisolver.nroots = 15
cas.kernel(mc.mo_coeff)
e = numpy.sort(numpy.atleast_1d(cas.e_tot))
print('CASCI levels (cm-1):')
print(numpy.array2string((e - e[0]) * 219474.63, precision=1, max_line_width=120))

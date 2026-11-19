#!/usr/bin/env python
'''
Gamma-point two-component (spinor) Hartree-Fock of a crystal.

The periodic SCF is the molecular spinor SCF on a pyscf Cell, with the Coulomb
interactions from the range-separated density fitting socutils.pbc.rsdf.RSDF:

    rsdf = RSDF(cell, auxbasis).build()
    mf = spinor_scf.periodic(spinor_hf.SCF(cell), rsdf)

and the usual shortcuts chain after it: .x2camf() is here the range-separated
X2C-1e of the cell plus the atomic mean-field spin-orbit (Gaunt, Breit) terms.
'''
import numpy
from pyscf.pbc import gto
from socutils.scf import spinor_hf
from socutils.pbc import spinor_scf
from socutils.pbc.rsdf import RSDF
from socutils.cd import accd

# LiH (rock salt), primitive cell
cell = gto.Cell()
cell.a = numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) * 4.083 / 2    # Angstrom
cell.atom = 'Li 0 0 0; H 2.0415 2.0415 2.0415'
cell.basis = 'cc-pvdz'
cell.exp_to_discard = 0.1       # diffuse functions are linearly dependent in a crystal
cell.nucmod = 'G'               # Gaussian nuclei, the model of the atomic X2CAMF code
cell.precision = 1e-10
cell.verbose = 4
cell.build()

# The density fitting: built once, used for J, K, V_ne and E_nn.  The auxiliary basis is generated from the
# orbital basis (atomic compact Cholesky, socutils.cd.accd); any pyscf auxiliary basis can be given instead.
# The nuclear model of RSDF must be the cell's.
auxbasis = accd.accd_basis(cell, verbose=0)
rsdf = RSDF(cell, auxbasis, nuc_zeta=spinor_scf.nuclear_zeta(cell))
rsdf.build()

# spinor HF with X2CAMF (Gaunt and Breit on by default, as for molecules)
mf = spinor_scf.periodic(spinor_hf.SCF(cell), rsdf).x2camf()
e = mf.kernel()
print('E(spinor HF, X2CAMF) = %.10f' % e)

# Kramers-restricted HF (needs zquatev): the same energy for a closed shell
mf_kr = spinor_scf.periodic(spinor_hf.KRHF(cell), rsdf).x2camf()
mf_kr.verbose = 0
print('E(Kramers-restricted) = %.10f' % mf_kr.kernel())

# without the two-electron picture-change terms
mf_dc = spinor_scf.periodic(spinor_hf.KRHF(cell), rsdf).x2camf(with_gaunt=False, with_breit=False)
mf_dc.verbose = 0
print('E(no Gaunt / Breit)   = %.10f' % mf_dc.kernel())

# The exchange G = 0 term: exxdiv = 'ewald' (default, as in pyscf) or None
mf_none = spinor_scf.periodic(spinor_hf.KRHF(cell), rsdf).x2camf()
mf_none.exxdiv = None
mf_none.verbose = 0
print('E(exxdiv = None)      = %.10f' % mf_none.kernel())

#!/usr/bin/env python
'''
A cell with a net charge.

Periodic Coulomb sums leave out the G = 0 term, which puts every cell in a
uniform neutralizing background.  A cell of net charge q then misses
q^2 nu_M / 2 (nu_M: the Madelung constant of the lattice; Makov and Payne,
first order).  socutils.pbc.charged.correct(mf) adds it to energy_nuc, for any
pyscf.pbc SCF object and for socutils' periodic spinor SCF.

He+ in cubic boxes: after the correction the cation has the finite-size error
of the neutral atom.
'''
from pyscf import gto, df
from socutils.scf import spinor_hf
from socutils.pbc import spinor_scf, charged
from socutils.pbc import coulomb_reference
from socutils.pbc.rsdf import RSDF

for charge in (1, 0):
    mol = gto.M(atom='He 0 0 0', basis='cc-pvdz', charge=charge, spin=charge, verbose=0)
    auxbasis = df.addons.aug_etb(mol, beta=2.0)
    mf_mol = spinor_hf.SCF(mol).x2camf().density_fit(auxbasis=auxbasis)
    e_mol = mf_mol.kernel()
    print('He, charge %d: molecular E = %.8f' % (charge, e_mol))
    for L in (12., 16., 20.):                                  # box edge, Bohr
        cell = coulomb_reference.molecule_in_box(mol, L)       # a cubic pyscf Cell around the molecule
        rsdf = RSDF(cell, auxbasis, precision=1e-11, nuc_zeta=1e14).build()    # 1e14: point nuclei
        mf = spinor_scf.periodic(spinor_hf.SCF(cell), rsdf).x2camf()
        mf.verbose = 0
        charged.correct(mf)                                    # no-op for the neutral cell
        e = mf.kernel()
        print('   L = %4.0f bohr: E - E(molecule) = %+.2e   (charge correction %.6f)'
              % (L, e - e_mol, mf.charge_correction))

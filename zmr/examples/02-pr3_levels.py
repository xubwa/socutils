#!/usr/bin/env python
'''
Pr3+ 4f^2 free-ion levels with multi-state UC- and PC-NEVPT2.

X2CAMF Hamiltonian, x2c-SVPall basis.  Orbitals: closed-shell Pr5+ KRHF as a
start, then the configuration-averaged HF (CAHF) of Pr3+ 4f^2 with socutils.
Model space: all 91 states of 4f^2 with equal weights; every J level is
exactly (2J+1)-fold degenerate.  With the complete active space as model
space the union-contracted PC space spans the whole target sectors, so
PC = UC.  Both E0 choices are run: state-specific E0 (the CASCI energy of
every model state) gives the level energies; the common E0 makes the
effective Hamiltonian invariant under any rotation of the model states but
enlarges all level energies by a fixed fraction of their CASCI values, the
norm of the first-order function (8.6 % here; docs/derivations/conventions.md
section 9).

Run on a full node (a few minutes, almost all of it the CAHF); socutils writes
an FCIDUMP file to the working directory.
'''
import time

import numpy
from pyscf import gto
from socutils.mcscf import zcahf, zmcscf
from socutils.scf import spinor_hf

from socutils.zmr.active import symmetry
from socutils.zmr.interface import socutils_x2c
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.uc.nevpt2 import UCNEVPT2

HARTREE2WAVENUMBER = 219474.6313632
basis = 'x2c-svpall'

# closed-shell Pr5+ as the starting point of the 4f^2 CAHF
mol5 = gto.M(atom='Pr 0 0 0', basis=basis, charge=5, verbose=0)
mf5 = spinor_hf.KRHF(mol5).x2camf().density_fit()
mf5.max_cycle = 200
mf5.conv_tol = 1e-10
mf5.kernel()

mol = gto.M(atom='Pr 0 0 0', basis=basis, charge=3, spin=2, verbose=0)
mf_df = spinor_hf.KRHF(mol).x2camf().density_fit()
mf_df.verbose = 0
mc = zmcscf.CASSCF(mf_df, 14, 2)
mc.fcisolver = zcahf.CAHF(mol)
mc.max_cycle_macro = 100
mc.conv_tol = 1e-10
mc.verbose = 0
mc.kernel(mo_coeff=mf5.mo_coeff)
print('CAHF energy %.10f, converged %s' % (mc.e_tot, mc.converged))

# exact integrals: 1s-4p frozen (36 spinors), 4d 5s 5p correlated (18), 4f active (14)
mf = spinor_hf.SCF(mol).x2camf()
mf.verbose = 0
eris = socutils_x2c.from_scf(mf, ncore=18, nact=14, nfrozen=36, mo_coeff=mc.mo_coeff, mol=mol)
# the 4f spinors of the CAHF are arbitrary mixtures within the j = 5/2 and 7/2
# shells; rotate them to eigenfunctions of one component of J (block diagonal
# sector Hamiltonians, same energies)
eris, sym = symmetry.adapt(eris)
print('active-space symmetry: %d generators, largest discarded integral %.1e Eh'
      % (sym.ngen, sym.error))


def levels(e, tol=1e-6):
    groups = []
    for x in numpy.sort(e):
        if groups and x - groups[-1][0] < tol:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def report(label, e):
    groups = levels(e)
    print('%s: degeneracies %s, largest spread within a level %.1e Eh' % (
        label, [len(g) for g in groups], max(max(g) - min(g) for g in groups)))
    print('    E - E(ground) / cm-1:',
          ' '.join('%.1f' % ((g[0] - groups[0][0]) * HARTREE2WAVENUMBER) for g in groups))


for e0 in ('state', 'average'):
    options = dict(nstate=91, e0=e0, method={'ccav': 'laplace', 'cavv': 'laplace'}, nquad=40)
    t0 = time.time()
    uc = UCNEVPT2(eris, 2, **options)
    uc.kernel()
    if e0 == 'state':
        report('CASCI', numpy.linalg.eigvalsh(uc.hmodel))
    report('UC-NEVPT2, e0=%s [%.0f s]' % (e0, time.time() - t0), uc.e_tot)
    t0 = time.time()
    pc = PCNEVPT2(eris, 2, contraction='union', **options)
    pc.kernel()
    report('PC-NEVPT2, e0=%s [%.0f s]' % (e0, time.time() - t0), pc.e_tot)

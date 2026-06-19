#!/usr/bin/env python
'''
Kramers-restricted spinor SCF.

spinor_hf.SCF is Kramers-unrestricted.  For a Kramers-restricted treatment use
spinor_hf.KRHF, which carries the same .x2camf()/.x2cmp() shortcuts but solves
the SCF with a Kramers-paired (quaternion) eigensolver, yielding exactly
doubly-degenerate Kramers pairs.

This path requires the bundled zquatev solver to be compiled
(run `make` at the repo root; see the install docs).
'''
from pyscf import gto
from socutils.scf import spinor_hf

mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

mf = spinor_hf.KRHF(mol).x2camf()
e = mf.kernel()
print('E(Kramers-restricted X2CAMF) = %.10f' % e)

"""DZ variant for the polarizability test: as make_y.py but with the most diffuse p primitive also
uncontracted (KEEP p: 1,2).  Written to dz_p1.py only (not GENBAS)."""
import os, sys
sys.path.insert(0, '../audit')
import dirac_basis
from socutils.tools.basisopt import contract, genbas
prims = {l: sorted(v, reverse=True) for l, v in dirac_basis.read('/home/xwang405/apps/DIRAC-23.0-Source/basis/dyall.v2z')[39].items()}
pattern = contract.pattern_occupied('Y', prims, {0: [2], 1: [1, 2], 2: [1, 2]}, gau_nuc=True)
e = contract.contract('Y', prims, pattern, kind='sf', gau_nuc=True, name='DYVDZ-SF-P1', conv=1e-9, max_iter=200)
genbas.write_pyscf('dz_p1.py', {'Y': genbas.to_pyscf(e)}, header='test variant: DYVDZ-SF + most diffuse p uncontracted')

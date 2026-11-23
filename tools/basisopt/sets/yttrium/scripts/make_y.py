"""Y:DYVDZ-SF - spin-free contracted Dyall double zeta for yttrium (requested by the user 2026-10-15).

No all-electron cc-pVDZ exists for Y (Basis Set Exchange: only cc-pVnZ-PP), so Dyall's dyall.v2z
primitives are used.  Contraction: spin-free X2CAMF atomic SCF (Gaussian nucleus), average of the
4d1 5s2 configuration; one contracted function per occupied shell.  Uncontracted, Dyall's 4d-block
DZ valence set (dyall_archive 4d_dz): 2nd s, 2nd p, 1st and 2nd d primitives (counted from the
smallest exponent); unoccupied l (f) uncontracted as in the file.
Checks: occupied orbital energies in the contracted space against the uncontracted primitives
(spin-free X2CAMF), smallest overlap eigenvalue per l."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'audit'))
import dirac_basis
from socutils.tools.basisopt import contract, genbas
D = '/home/xwang405/apps/DIRAC-23.0-Source/basis/dyall.v2z'
prims = {l: sorted(v, reverse=True) for l, v in dirac_basis.read(D)[39].items()}
print('dyall.v2z Y:', ' '.join('%d%s' % (len(prims[l]), 'spdfg'[l]) for l in sorted(prims)))
for l in sorted(prims):
    print('   %s: smallest %s' % ('spdfg'[l], ' '.join('%.5g' % e for e in sorted(prims[l])[:4])))
KEEP = {0: [2], 1: [2], 2: [1, 2]}
pattern = contract.pattern_occupied('Y', prims, KEEP, gau_nuc=True)
e = contract.contract('Y', prims, pattern, kind='sf', gau_nuc=True, name='DYVDZ-SF', conv=1e-9, max_iter=200)
nfun = sum((2 * l + 1) * c.shape[1] for l, (x, c) in e.shells.items())
print('Y:DYVDZ-SF  %s -> %s  (%d functions); contracted per l: %s' % (
    ''.join('%d%s' % (len(x), 'spdfg'[l]) for l, (x, c) in sorted(e.shells.items())),
    ''.join('%d%s' % (c.shape[1], 'spdfg'[l]) for l, (x, c) in sorted(e.shells.items())), nfun,
    {('spdfg'[l]): pattern[l][0] for l in sorted(pattern) if pattern[l][0]}))
for l, (x, c) in sorted(e.shells.items()):
    a = x[:, None]; b = x[None, :]
    s = (2 * np.sqrt(a * b) / (a + b)) ** (l + 1.5)
    cc = c / np.sqrt(np.einsum('pi,pq,qi->i', c, s, c))
    print('   %s: smallest overlap eigenvalue %.1e' % ('spdfg'[l], np.linalg.eigvalsh(cc.T @ s @ cc)[0]))
rows = contract.contraction_error(e, 'Y', 'sfx2c1e', gau_nuc=True)
for n, l, tj, x0, x1 in rows:
    print('   %d%s  uncontracted %.6f  contracted %.6f  diff %.1e' % (n, 'spdfg'[l], x0, x1, x1 - x0))
out = os.path.join(HERE, 'GENBAS.y')
if os.path.exists(out): os.remove(out)
e.name = 'Y:DYVDZ-SF'
genbas.append(out, e)
genbas.write_pyscf(os.path.join(HERE, 'dyvdz_sf.py'), {'Y': genbas.to_pyscf(e)},
                   header='Y:DYVDZ-SF: dyall.v2z, spin-free X2CAMF contraction (4d1 5s2 average), Gaussian nucleus (draft of 2026-10-15)')
print('written:', out, os.path.join(HERE, 'dyvdz_sf.py'))

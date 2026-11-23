"""Y:DYVTZ-SF-T05 (requested by the Eu:YSO session, 2026-11-10): dyall.v3z without the 4s4p
outer-core f 1.115 and without every primitive with exponent < 0.05 (the Eu:YSO cell rule against
linear dependence; Y is Y3+ there).  Spin-free X2CAMF contraction (4d1 5s2 average, Gaussian
nucleus), one contracted function per occupied shell; uncontracted: Dyall 4d_tz valence rule
(1st+3rd s, 2nd+3rd p, 1st-3rd d) counted from the smallest REMAINING exponent; f, g uncontracted.
Checks against the full uncontracted dyall.v3z: neutral-Y orbital energies (contraction error in
the truncated primitives + truncation error of the primitives), Y+ / Y3+ RHF energies and
finite-field polarizabilities."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'audit'))
import dirac_basis
from socutils.tools.basisopt import contract, genbas, engine
full = {l: sorted(v, reverse=True) for l, v in dirac_basis.read('/home/xwang405/apps/DIRAC-23.0-Source/basis/dyall.v3z')[39].items()}
prims = {l: [e for e in v if e >= 0.05 and not (l == 3 and abs(e / 1.1153443 - 1) < 1e-6)] for l, v in full.items()}
for l in sorted(full):
    gone = sorted(set(full[l]) - set(prims[l]))
    print('%s: %d -> %d  removed %s' % ('spdfg'[l], len(full[l]), len(prims[l]), ' '.join('%.4g' % e for e in gone)))
pattern = contract.pattern_occupied('Y', prims, {0: [1, 3], 1: [2, 3], 2: [1, 2, 3]}, gau_nuc=True)
e = contract.contract('Y', prims, pattern, kind='sf', gau_nuc=True, name='DYVTZ-SF-T05', conv=1e-9, max_iter=300)
nfun = sum((2 * l + 1) * c.shape[1] for l, (x, c) in e.shells.items())
print('== Y:DYVTZ-SF-T05  %s -> %s  (%d functions)' % (''.join('%d%s' % (len(x), 'spdfg'[l]) for l, (x, c) in sorted(e.shells.items())),
      ''.join('%d%s' % (c.shape[1], 'spdfg'[l]) for l, (x, c) in sorted(e.shells.items())), nfun))
for l, (x, c) in sorted(e.shells.items()):
    n, unc = pattern.get(l, (0, []))
    print('   %s: %s' % ('spdfg'[l], ('free ' + ' '.join('%.5g' % x[i] for i in unc)) if n else 'all uncontracted: ' + ' '.join('%.4g' % v for v in x)))
# orbital energies: full uncontracted v3z vs truncated primitives (same AOC SCF), and contraction error
ef = engine.atom('Y', full, 'sfx2c1e', gau_nuc=True, aoc=True, max_iter=300)
et = engine.atom('Y', prims, 'sfx2c1e', gau_nuc=True, aoc=True, max_iter=300)
print('   AOC SCF energy: full v3z %.8f  truncated prims %.8f  diff %.2e' % (ef.e_scf, et.e_scf, et.e_scf - ef.e_scf))
of = {(n, l): x for n, l, tj, f, x in ef.occupied()}
ot = {(n, l): x for n, l, tj, f, x in et.occupied()}
seen = set()
print('   orbital  full-v3z       truncation   contraction   total')
for n, l, tj, x0, x1 in contract.contraction_error(e, 'Y', 'sfx2c1e', gau_nuc=True):
    if (n, l) in seen: continue
    seen.add((n, l))
    tr = ot[(n, l)] - of[(n, l)]
    print('   %d%s  %12.6f  %10.1e  %10.1e  %10.1e' % (n, 'spdfg'[l], of[(n, l)], tr, x1 - x0, tr + x1 - x0))
out = os.path.join(HERE, 'GENBAS.y')
e.name = 'Y:DYVTZ-SF-T05'
if e.name not in [l.strip() for l in open(out) if l.startswith('Y:')]:
    genbas.append(out, e)
genbas.write_pyscf(os.path.join(HERE, 'dyvtz_sf_t05.py'), {'Y': genbas.to_pyscf(e)},
                   header='Y:DYVTZ-SF-T05: dyall.v3z without f 1.115 and without exponents < 0.05; spin-free X2CAMF '
                          'contraction (4d1 5s2 average), Gaussian nucleus; Dyall 4d_tz valence rule. Source: y_sf/make_y_t05.py')
genbas.write_pyscf(os.path.join(HERE, 'dyall_v3z_unc.py'), {'Y': [[l, [x, 1.0]] for l in sorted(full) for x in full[l]]},
                   header='full dyall.v3z Y, uncontracted (reference for the checks)')

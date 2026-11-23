"""U entries with correlating functions at the 6d radius (for the 6d^2 pair of U-, the 6d state of
the EA calculation): on top of the -P1 entries (GENBAS.thuof_ea), TZ + one g and one h, QZ + one h,
exponents a = (l + 1) / (2 r_max^2) with r_max = 2.465 bohr, the radial maximum of the 6d3/2 spinor
of U- 5f3 6d2 7s2 (X2CAMF + Gaunt, configuration average; d6pair/r6d.out): g 0.411, h 0.494.
Reason: Dyall's U sets have no g or h near 0.4-0.5 at TZ (g 0.175 -> 0.824, h 0.265 -> 1.54) and no
h there at QZ (0.193 -> 0.834), while his Th sets (ground state 6d2 7s2) have g 0.494 (TZ) and
h 0.51 (QZ) at that radius; the 6d radius is the same in Th and U (2.38 bohr) and 4 % larger in U-.
Uncontracted, shared by both kappas.  Names with "-6D"."""
import os
import numpy as np
from socutils.tools.basisopt import genbas
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'make_thuof.py')).read()
exec(src[src.index('THR = 1e-5'):src.index('def drop_redundant(')])       # LN, min_eigs
RMAX = 2.465
ADD = {'T': {4: [5 / (2 * RMAX ** 2)], 5: [6 / (2 * RMAX ** 2)]}, 'Q': {5: [6 / (2 * RMAX ** 2)]}}
out = os.path.join(HERE, 'GENBAS.thuof_ea6d')
if os.path.exists(out): os.remove(out)
py = {}
for kind in ('DYV', 'ADYV', 'DYCV', 'ADYCV'):
    for z in 'TQ':
        ent = genbas.read(os.path.join(HERE, 'GENBAS.thuof_ea'), 'U:%s%sZ-P1-SOG' % (kind, z))
        shells = dict(ent.shells)
        notes = []
        for l, xs in ADD[z].items():
            ex, c = shells[l]
            nb = np.sort(ex)
            for x in xs:
                lo = nb[nb < x].max() if (nb < x).any() else None
                hi = nb[nb > x].min() if (nb > x).any() else None
                notes.append('%s %.4f (neighbours %s / %s, ratios %.2f / %.2f)' % (LN[l], x, '%.4g' % lo if lo else '-', '%.4g' % hi if hi else '-',
                             x / lo if lo else 0, hi / x if hi else 0))
                ex = np.concatenate([ex, [x]])
                c = np.hstack([np.vstack([c, np.zeros((1, c.shape[1]))]), np.eye(len(ex))[:, [-1]]])
            shells[l] = (ex, c)
        new = genbas.Entry('U:%s%sZ-P1-6D-SOG' % (kind, z), ent.comment, shells)
        m0, m1 = min_eigs(ent), min_eigs(new)
        print('%-22s + %s;  smallest overlap eigenvalue %.1e -> %.1e (g %.1e, h %.1e)' % (new.name, '; '.join(notes), min(m0.values()), min(m1.values()),
              m1.get((4, 9), np.nan), m1.get((5, 11), np.nan)), flush=True)
        genbas.append(out, new)
        py.setdefault(new.name.split(':')[1], {})['U'] = genbas.to_pyscf(new, kappa=True)
for name, d in py.items():
    genbas.write_pyscf(os.path.join(HERE, 'pyscf', name.lower().replace('-', '_') + '.py'), d,
                       header='%s: -P1 entry plus g/h correlating functions at the 6d radius of U- (draft of 2026-10-15)' % name)
print('written:', out)

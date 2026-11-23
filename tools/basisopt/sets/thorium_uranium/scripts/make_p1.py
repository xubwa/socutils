"""U entries for the electron-affinity calculation of the zccsdt session (user's decision,
2026-10-15): Dyall's valence uncontraction leaves the most diffuse p primitive of the SCF set
inside the contracted 2p-6p functions (DZ: p 2nd and 4th free, TZ 2,3,5,6, QZ 2,3,4,6,7,8); the
7p of U- then has to do without it, and the contraction error of EA_SCF(7p1/2) was -9.8 meV at TZ
(diffuse set present).  Freeing that primitive (shared by both kappas) brings it to +0.07 meV and
leaves everything else unchanged (ea_scf/ea_p1free_T.res).  Applied here to the valence sets
(DYVnZ-SOG, ADYVnZ-SOG) and to version 3 (union) of the core-valence sets (DYCVnZ-SOG, ADYCVnZ-SOG):
names with "-P1".  A patch of Dyall's rule for this calculation, not the final rule (the outer
exponents being denser than the zeta level is the underlying reason)."""
import os
import numpy as np
from socutils.tools.basisopt import genbas
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'make_thuof.py')).read()
exec(src[src.index('THR = 1e-5'):src.index('def drop_redundant(')])       # min_eigs
SRC = {'DYV': 'GENBAS.thuof', 'ADYV': 'GENBAS.thuof_aug', 'DYCV': 'GENBAS.thuof_cv3', 'ADYCV': 'GENBAS.thuof_cv3'}
out = os.path.join(HERE, 'GENBAS.thuof_ea')
if os.path.exists(out): os.remove(out)
py = {}
for kind in ('DYV', 'ADYV', 'DYCV', 'ADYCV'):
    for z in 'DTQ':
        ent = genbas.read(os.path.join(HERE, SRC[kind]), 'U:%s%sZ-SOG' % (kind, z))
        shells = dict(ent.shells)
        ex, c = shells[1]
        aug = kind.startswith('A')
        scf = np.sort(ex[:-1] if aug else ex)                 # the SCF primitives (aug: last row is the added diffuse p)
        i = int(np.argmin(abs(ex - scf[0])))
        unit = [j for j in range(c.shape[1]) if (abs(c[:, j]) > 1e-12).sum() == 1 and abs(c[i, j]) > 0]
        if unit:
            print('U:%s%sZ-SOG: p %.5g already uncontracted, unchanged' % (kind, z, ex[i])); new = ent
        else:
            col = np.zeros((len(ex), 1)); col[i] = 1.0
            shells[1] = (ex, np.hstack([c, col]))
            new = genbas.Entry('U:%s%sZ-P1-SOG' % (kind, z), ent.comment, shells)
        m0, m1 = min_eigs(ent), min_eigs(new)
        print('%-20s freed p %.5g;  info line %r;  smallest overlap eigenvalue p1/2 %.1e -> %.1e, p3/2 %.1e -> %.1e, overall %.1e -> %.1e' % (
            new.name, ex[i], new.comment, m0[(1, 1)], m1[(1, 1)], m0[(1, 3)], m1[(1, 3)], min(m0.values()), min(m1.values())), flush=True)
        genbas.append(out, new)
        py.setdefault(new.name.split(':')[1], {})['U'] = genbas.to_pyscf(new, kappa=True)
for name, d in py.items():
    genbas.write_pyscf(os.path.join(HERE, 'pyscf', name.lower().replace('-', '_') + '.py'), d,
                       header='%s: as without -P1, plus the most diffuse p primitive uncontracted (for the U EA calculation; draft of 2026-10-15)' % name)
print('written:', out)

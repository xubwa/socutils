"""Near-HF-limit SF-X2C-1e reference for Ca 4s2 (Gaussian nucleus): even-tempered s/p sets with
decreasing ratio until the energy is stable; then SCF errors of the candidate primitive sets."""
import numpy as np
from socutils.tools.basisopt import engine
exec(open('ca_scf.py').read().split('for name in')[0])
def et(lo, hi, r):
    n = int(np.ceil(np.log(hi / lo) / np.log(r))) + 1
    return list(lo * r ** np.arange(n))[::-1]
REFS = []
for r in (2.2, 2.0, 1.85, 1.75):
    p = {0: et(0.008, 2e6, r), 1: et(0.008, 2e5, r), 2: et(0.03, 50, r * 1.2)}
    res = engine.atom('Ca', p, 'sfx2c1e', gau_nuc=True)
    REFS.append(res.e_scf); print('even-tempered ratio %.2f  %ds %dp  E = %.8f %s' % (r, len(p[0]), len(p[1]), res.e_scf, res.converged))
ref = min(x for x in REFS if x == x)
for name in ['cc-pVTZ', 'cc-pVQZ', 'cc-pV5Z', 'cc-pVTZ-X2C', 'cc-pVQZ-X2C', 'dyall-v4z']:
    e = engine.atom('Ca', prims(name), 'sfx2c1e', gau_nuc=True).e_scf
    print('%-12s dE = %8.4f mEh' % (name, (e - ref) * 1e3))

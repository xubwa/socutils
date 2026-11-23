"""Ca aug-cc-pV{T,Q,5}Z X2C sets, scalar (sf) and j-adapted (sog), Gaussian nucleus.
TZ/QZ: primitives of aug-cc-pV{T,Q}Z-X2C (Hill & Peterson).  5Z: s/p primitives of cc-pVQZ-X2C
(SF-X2C HF error ~0 against dyall-v4z) with 4 outer s/p free, d/f/g/h of cc-pV5Z (Koput &
Peterson), aug functions at ratio 2.5 as in Hill & Peterson.
Contraction (user decisions 2026-11-07): only occupied atomic orbitals are contracted
(1s-4s, 2p-3p); the unoccupied 4p contraction of the source is replaced by one more free p
primitive in the 4p region; d stays fully uncontracted."""
import numpy as np
from socutils.tools.basisopt import contract, genbas
from src import src
def build(z):
    if z in 'TQ':
        prims, pat = src('aug-cc-pV%sZ-X2C' % z)
    else:
        q, qp = src('cc-pVQZ-X2C'); f, _ = src('cc-pV5Z')
        prims = {0: q[0], 1: q[1], 2: f[2], 3: f[3], 4: f[4], 5: f[5]}
        pat = {0: (4, [26, 27, 28, 29]), 1: (3, [16, 17, 18, 19])}
        for l in (0, 1, 2, 3, 4, 5):          # aug
            prims[l] = prims[l] + [prims[l][-1] / 2.5]
        pat[0] = (4, pat[0][1] + [len(prims[0]) - 1]); pat[1] = (3, pat[1][1] + [len(prims[1]) - 1])
    n, free = pat[1]
    assert n == 3
    extra = max(i for i in range(len(prims[1])) if i not in free and i < max(free))
    pat[1] = (2, sorted(free + [extra]))
    for l in prims:
        if l >= 2: pat[l] = (0, list(range(len(prims[l]))))
    return prims, pat, prims[1][extra]
for z in 'TQ5':
    prims, pat, ex = build(z)
    print('== Ca aug-cc-pV%sZ: %s; extra free p %.5g' % (z, ' '.join('%d%s' % (len(prims[l]), 'spdfgh'[l]) for l in sorted(prims)), ex))
    for kind in ('sf', 'sog'):
        e = contract.contract('Ca', prims, pat, kind=kind, aoc=True, gau_nuc=True, max_iter=1000, name='APV%sZ-%s' % (z, kind.upper()))
        genbas.append('GENBAS.canh', e)
        print('  %-4s E_SCF %.8f  functions %s' % (kind, e.scf.e_scf, ' '.join('%d%s' % (e.shells[l][1].shape[1], 'spdfgh'[l]) for l in sorted(e.shells))))

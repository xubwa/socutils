"""LN:DYVTZ-SF-T07 for Ln3+ in Y2SiO5 (requested by the Eu:YSO session 2026-11-16).

Primitives: dyall.v3z; every primitive with exponent < 0.07 removed (periodic decontracted metric, as
Y:DYVTZ-SF-T07); all tight s/p kept.  Contraction: spin-free X2CAMF atomic SCF of the TRIVALENT ion,
configuration average of 4f^n (n electrons spread over the 14 4f spinors), Gaussian nucleus; one
contracted function per occupied shell (1s-5s, 2p-5p, 3d-4d, 4f).  Uncontracted (counted from the
smallest REMAINING exponent): Dyall's 4f_tz combined valence set (all except La): s 1,3,4,6; p 1,3,5,6;
d 1-5; f 1,3,4,5; g, h (unoccupied) uncontracted.
usage: KIND=sf|sog make_ln.py Pr Yb ...  (sog: j-adapted, X2CAMF DC-Gaunt AMF, 4f5/2 and 4f7/2 contracted separately)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'audit'))
import dirac_basis
from socutils.tools.basisopt import contract, genbas, engine
from pyscf.data.elements import charge
CUT = 0.07
NF = {'Ce': 1, 'Pr': 2, 'Nd': 3, 'Sm': 5, 'Eu': 6, 'Gd': 7, 'Tb': 8, 'Dy': 9, 'Ho': 10, 'Er': 11, 'Tm': 12, 'Yb': 13}
KEEP = {0: [1, 3, 4, 6], 1: [1, 3, 5, 6], 2: [1, 2, 3, 4, 5], 3: [1, 3, 4, 5]}
D = '/home/xwang405/apps/DIRAC-23.0-Source/basis/dyall.v3z'
def occ_ion(n):          # s, p1/2, p3/2, d3/2, d5/2, f5/2, f7/2 for [Kr]4d10 5s2 5p6 4f^n
    return [10, 8, 16, 8, 12, n * 6 / 14, n * 8 / 14]
def build(el):
    full = {l: sorted(v, reverse=True) for l, v in dirac_basis.read(D)[charge(el)].items()}
    prims = {l: [e for e in v if e >= CUT] for l, v in full.items()}
    nocc = {0: 5, 1: 4, 2: 2, 3: 1}
    pattern = {}
    for l, ex in prims.items():
        if l in nocc:
            n = len(ex)
            pattern[l] = (nocc[l], sorted(n - k for k in KEEP[l]))
        else:
            pattern[l] = (0, list(range(len(ex))))
    return full, prims, pattern
if __name__ == '__main__':
    KIND = os.environ.get('KIND', 'sf')
    out = os.path.join(HERE, 'GENBAS.ln')
    for el in sys.argv[1:]:
        full, prims, pattern = build(el)
        print('== %s3+ 4f%d   dyall.v3z %s' % (el, NF[el], ' '.join('%d%s' % (len(full[l]), 'spdfgh'[l]) for l in sorted(full))))
        for l in sorted(full):
            gone = sorted(set(full[l]) - set(prims[l]))
            if gone: print('   %s removed %s' % ('spdfgh'[l], ' '.join('%.4g' % e for e in gone)))
        name = 'DYVTZ-%s-T07' % KIND.upper()
        e = contract.contract(el, prims, pattern, kind=KIND, gau_nuc=True, aoc=True, occ=occ_ion(NF[el]),
                              name=name, conv=1e-9, max_iter=1000)
        nfun = sum((2 * l + 1) * c.shape[1] for l, (x, c) in e.shells.items())
        print('   %s -> %s (%d functions); AOC E_SCF in primitives %.8f' % (
            ''.join('%d%s' % (len(x), 'spdfgh'[l]) for l, (x, c) in sorted(e.shells.items())),
            ''.join('%d%s' % (c.shape[1], 'spdfgh'[l]) for l, (x, c) in sorted(e.shells.items())), nfun, e.scf.e_scf))
        for l, (x, c) in sorted(e.shells.items()):
            n, unc = pattern[l]
            print('   %s free %s' % ('spdfgh'[l], ' '.join('%.4g' % x[i] for i in unc)))
        e.name = '%s:%s' % (el.upper(), name)
        if not os.path.exists(out) or e.name not in [s.strip() for s in open(out)]:
            genbas.append(out, e)
        genbas.write_pyscf(os.path.join(HERE, '%s_dyvtz_%s_t07.py' % (el.lower(), KIND)), {el: genbas.to_pyscf(e, kappa=(KIND != 'sf'))},
                           header=('%s: dyall.v3z without exponents < %.2f; ' + ('spin-free X2CAMF' if KIND == 'sf' else 'j-adapted X2CAMF (DC-Gaunt AMF), kappa format,') + ' contraction of %s3+ 4f%d (configuration average), '
                                  'Gaussian nucleus; Dyall 4f_tz combined valence uncontraction. Source: lnbas/make_ln.py') % (e.name, CUT, el, NF[el]))
        genbas.write_pyscf(os.path.join(HERE, '%s_v3z_unc.py' % el.lower()), {el: [[l, [x, 1.0]] for l in sorted(full) for x in full[l]]},
                           header='full dyall.v3z %s uncontracted (reference for the checks)' % el)

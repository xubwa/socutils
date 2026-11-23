"""Core-valence (5s5p5d) U entries U:DYCV{D,T,Q}Z-SOG and their diffuse-augmented versions
U:ADYCV{D,T,Q}Z-SOG, following Dyall's 5f-block archive, set 2 "valence + 5d correlation":
on top of the valence uncontraction of U:DYVnZ-SOG, uncontract (counted from the smallest
exponent of the SCF set)
    DZ: 6th s, 6th p, 6th d, 5th f
    TZ: 7th, 9th s; 8th, 9th p; 6th, 7th d; 4th, 5th f
    QZ: 9th, 10th, 11th s; 9th, 10th, 11th p; 7th d; 5th, 6th f
no extra g/h (Dyall: covered by the 5f correlating set).  Same j-adapted X2CAMF + Gaunt
contraction (Gaussian nucleus) and the same redundancy rule as make_thuof.py (a contracted shell
whose overlap eigenvalue falls below 1e-5 is dropped).  Diffuse: one even-tempered primitive per
l as in make_aug.py.

Check (scalar proxy, the same uncontraction patterns with spin-free contractions): U6+ ([Rn],
closed shell) RHF + MP2, sf-X2C-1e, Gaussian nucleus, 5s5p5d6s6p correlated (30 orbitals frozen),
recovered fraction of the uncontracted dyall.vnz MP2 energy for: the valence pattern, the
valence pattern + the extra g/h of CFOUR's U:DYPCVnZ-SO, Dyall's core-valence pattern, and that
pattern + the CFOUR g/h."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'audit'))
import dirac_basis
from socutils.tools.basisopt import contract, genbas
src = open(os.path.join(HERE, 'make_thuof.py')).read()
exec(src[src.index('THR = 1e-5'):src.index('def check(')])           # THR, LN, KEEP_5F, min_eigs, drop_redundant
def ext(exps, k=1):
    a = np.sort(np.asarray(exps)); r = a[1] / a[0]
    return [a[0] / r ** i for i in range(1, k + 1)]
def augment(entry, name):
    shells = {}
    for l, (ex, c) in entry.shells.items():
        new = ext(ex)
        shells[l] = (np.concatenate([ex, new]), np.hstack([np.vstack([c, np.zeros((1, c.shape[1]))]), np.vstack([np.zeros((len(ex), 1)), np.eye(1)])]))
    return genbas.Entry(name, entry.comment, shells)
D = '/home/xwang405/apps/DIRAC-23.0-Source/basis/'
CV = {2: {0: [6], 1: [6], 2: [6], 3: [5]},
      3: {0: [7, 9], 1: [8, 9], 2: [6, 7], 3: [4, 5]},
      4: {0: [9, 10, 11], 1: [9, 10, 11], 2: [7], 3: [5, 6]}}
CFOUR_EXTRA = {3: {4: [3.2534]}, 4: {3: [10.72], 5: [3.9024]}}     # what U:DYPCVnZ-SO adds (TZ g; QZ f, h)
out = os.path.join(HERE, 'GENBAS.thuof_cv')
if os.path.exists(out): os.remove(out)
py = {}
for n, zl in ((2, 'D'), (3, 'T'), (4, 'Q')):
    d = dirac_basis.read(D + 'dyall.v%dz' % n)[92]
    prims = {l: sorted(v, reverse=True) for l, v in d.items()}
    keep = {l: sorted(set(KEEP_5F[n].get(l, []) + CV[n].get(l, []))) for l in set(KEEP_5F[n]) | set(CV[n])}
    pattern = contract.pattern_occupied('U', prims, keep, gau_nuc=True)
    e = contract.contract('U', prims, pattern, kind='sog', gau_nuc=True, name='DYCV%sZ-SOG' % zl, conv=2e-8, max_iter=200)
    before = min(min_eigs(e).values())
    dropped = drop_redundant(e)
    me = min_eigs(e)
    so = e.so_info()
    nfun = sum((2 * l + 1) * (c.shape[1] - so.get(l, 0)) for l, (x, c) in e.shells.items())
    print('U:DYCV%sZ-SOG  uncontracted (from smallest) %s;  info line %r;  dropped %s;  %d functions' % (
        zl, {LN[l]: v for l, v in sorted(keep.items())}, e.comment, ', '.join(dropped) or '-', nfun))
    print('   freed core-correlating exponents: %s' % '  '.join('%s: %s' % (LN[l], ' '.join('%.4g' % sorted(prims[l])[k - 1] for k in v)) for l, v in sorted(CV[n].items())))
    print('   smallest overlap eigenvalue before dropping %.1e, after %.1e (%s)' % (before, min(me.values()), min(me, key=me.get)))
    err = max(abs(b - a) for *_, a, b in contract.contraction_error(e, 'U', 'x2camf-g', gau_nuc=True))
    print('   largest occupied spinor-energy error against the uncontracted primitives %.1e Eh' % err, flush=True)
    a = augment(e, 'U:ADYCV%sZ-SOG' % zl)
    ma = min_eigs(a)
    print('U:ADYCV%sZ-SOG  + diffuse %s;  smallest overlap eigenvalue %.1e (%s)' % (zl, ' '.join('%s %.4g' % (LN[l], x[-1]) for l, (x, c) in sorted(a.shells.items())), min(ma.values()), min(ma, key=ma.get)), flush=True)
    for ent in (e, a):
        genbas.append(out, ent)
        py.setdefault(ent.name.split(':')[1], {})['U'] = genbas.to_pyscf(ent, kappa=True)

    # scalar proxy check on U6+
    from pyscf import gto, scf, mp
    def run(basis):
        m = gto.M(atom='U', basis={'U': basis}, charge=6, verbose=0, nucmod='G')
        f = scf.RHF(m).sfx2c1e(); f.conv_tol = 1e-10; f.max_cycle = 200; f.kernel()
        return f.e_tot, mp.MP2(f, frozen=30).kernel()[0], mp.MP2(f, frozen=39).kernel()[0], m.nao_nr()
    ref = run([[l, [x, 1.0]] for l in sorted(prims) for x in prims[l]])
    def sf(keep_, extra=None):
        pat = contract.pattern_occupied('U', prims, keep_, gau_nuc=True)
        ent = contract.contract('U', prims, pat, kind='sf', gau_nuc=True, name='X', conv=2e-8, max_iter=200)
        b = genbas.to_pyscf(ent)
        for l, xs in (extra or {}).items():
            b += [[l, [x, 1.0]] for x in xs]
        return b
    print('   U6+ scalar MP2 (5s5p5d6s6p | 6s6p correlated), uncontracted: %d functions, E2 %.2f | %.2f mEh' % (ref[3], ref[1] * 1e3, ref[2] * 1e3))
    for lab, b in (('valence pattern (DYVnZ)', sf(KEEP_5F[n])), ('valence + CFOUR g/h', sf(KEEP_5F[n], CFOUR_EXTRA.get(n))),
                   ('Dyall core-valence (DYCVnZ)', sf(keep)), ('Dyall core-valence + CFOUR g/h', sf(keep, CFOUR_EXTRA.get(n)))):
        r = run(b)
        print('      %-32s %3d functions  dE(HF) %+.2f mEh  recovered %.4f | %.4f' % (lab, r[3], (r[0] - ref[0]) * 1e3, r[1] / ref[1], r[2] / ref[2]), flush=True)
for name, dd in py.items():
    genbas.write_pyscf(os.path.join(HERE, 'pyscf', name.lower().replace('-', '_') + '.py'), dd,
                       header='%s: Dyall valence + 5d correlating uncontraction, j-adapted X2CAMF + Gaunt (draft of 2026-10-14)' % name)
print('written:', out)

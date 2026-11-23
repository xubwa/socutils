"""Diffuse-augmented U entries U:ADYV{D,T,Q}Z-SOG (requested by the user via the zccsdt session,
2026-10-14, for the electron affinity of U: anion states with the extra electron in 7p1/2 or 6d).

Construction: the U:DYVnZ-SOG entries of GENBAS.thuof unchanged (same primitives, same j-adapted
contractions, same uncontracted functions) plus ONE diffuse primitive per angular momentum
present, uncontracted and shared by both kappas, its exponent the even-tempered extension of
the two most diffuse exponents of that l: a = a1^2 / a2 (the convention of Peterson's
aug-cc-pVnZ sets for transition metals and actinides; Dyall's archive has no diffuse functions
for the 5f block and DIRAC's dyall.avnz sets do not contain the actinides).  Zero coefficients
of the new primitive in the contracted functions (as for aug-cc-pVnZ).

Checks: smallest overlap eigenvalue per l and kappa; atomic SCF (X2CAMF + Gaunt, Gaussian
nucleus, fractional occupation of the open shells) of U 5f3 6d1 7s2, U- 5f3 6d1 7s2 7p1/2 and
U- 5f3 6d2 7s2 in the uncontracted primitives without diffuse, with one diffuse per l (this
set) and with two (a second even-tempered step: reference); spinor energies of the occupied
shells of each state in the contracted space of the entry against its uncontracted primitives."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from socutils.tools.basisopt import engine, contract, genbas
import importlib.util
spec = importlib.util.spec_from_file_location('mt', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_thuof.py'))
HERE = os.path.dirname(os.path.abspath(__file__))
LN = 'spdfghik'
src = os.path.join(HERE, 'GENBAS.thuof')

def min_eigs(entry):                                       # as in make_thuof.py
    so, out = entry.so_info(), {}
    for l, (ex, c) in entry.shells.items():
        a, b = ex[:, None], ex[None, :]
        s = (2 * np.sqrt(a * b) / (a + b)) ** (l + 1.5)
        n = so.get(l, 0)
        blocks = [(2 * l + 1, list(range(c.shape[1])))] if (l == 0 or n == 0) else [
            (2 * l - 1, list(range(0, n)) + list(range(2 * n, c.shape[1]))), (2 * l + 1, list(range(n, c.shape[1])))]
        for tj, cols in blocks:
            cc = c[:, cols]; cc = cc / np.sqrt(np.einsum('pi,pq,qi->i', cc, s, cc))
            out[(l, tj)] = float(np.linalg.eigvalsh(cc.T @ s @ cc)[0])
    return out

def ext(exps, k=1):
    a = np.sort(np.asarray(exps))
    r = a[1] / a[0]
    return [a[0] / r ** i for i in range(1, k + 1)]

def augment(entry, name, k=1):
    shells = {}
    for l, (ex, c) in entry.shells.items():
        new = ext(ex, k)
        ex2 = np.concatenate([ex, new])
        c2 = np.vstack([c, np.zeros((k, c.shape[1]))])
        c2 = np.hstack([c2, np.vstack([np.zeros((len(ex), k)), np.eye(k)])])
        shells[l] = (ex2, c2)
    e = genbas.Entry(name, entry.comment, shells)
    return e

# occupations per kappa block (s, p1/2, p3/2, d3/2, d5/2, f5/2, f7/2); open shells spread by 2j+1
def occ(nd, n7p):
    return [14, 10 + n7p, 20, 12 + nd * 4 / 10, 18 + nd * 6 / 10, 6 + 3 * 6 / 14, 8 + 3 * 8 / 14]
STATES = [('U  5f3 6d1 7s2', occ(1, 0)), ('U- 5f3 6d1 7s2 7p1/2', occ(1, 1)), ('U- 5f3 6d2 7s2', occ(2, 0))]
EV = 27.211386

out = os.path.join(HERE, 'GENBAS.thuof_aug')
if os.path.exists(out): os.remove(out)
py = {}
for z in 'DTQ':
    base = genbas.read(src, 'U:DYV%sZ-SOG' % z)
    aug = augment(base, 'U:ADYV%sZ-SOG' % z)
    print('U:ADYV%sZ-SOG  info line %r (unchanged)' % (z, aug.comment))
    for l, (ex, c) in sorted(aug.shells.items()):
        print('   %s: added %.6g (two most diffuse before: %s; ratio %.2f)' % (LN[l], ex[-1], ' '.join('%.5g' % x for x in sorted(base.shells[l][0])[:2]),
              sorted(base.shells[l][0])[1] / sorted(base.shells[l][0])[0]))
    m0, m1 = min_eigs(base), min_eigs(aug)
    print('   smallest overlap eigenvalue: before %.1e (%s), after %.1e (%s)' % (min(m0.values()), min(m0, key=m0.get), min(m1.values()), min(m1, key=m1.get)))
    for (l, tj) in sorted(m1):
        if m1[(l, tj)] < 1e-3:
            print('      %s%d/2: %.1e -> %.1e' % (LN[l], tj, m0[(l, tj)], m1[(l, tj)]))
    genbas.append(out, aug)
    py.setdefault('ADYV%sZ-SOG' % z, {})['U'] = genbas.to_pyscf(aug, kappa=True)
    # atomic checks
    prim = {k: {l: sorted(ex, reverse=True) for l, (ex, c) in e.shells.items()} for k, e in (('none', base), ('one', aug))}
    prim['two'] = {l: sorted(list(ex) + ext(ex, 1)[:0] + [ext(ex, 2)[1]], reverse=True) for l, (ex, c) in aug.shells.items()}
    prim['two'] = {l: sorted(list(base.shells[l][0]) + ext(base.shells[l][0], 2), reverse=True) for l in base.shells}
    E = {}
    for key, p in prim.items():
        for lab, o in STATES:
            r = engine.atom('U', p, 'x2camf-g', occ=o, gau_nuc=True, conv=1e-9, max_iter=300)
            E[(key, lab)] = r
    print('   atomic SCF (fractional occupation), electron affinity E(U) - E(U-) in eV:')
    for key in ('none', 'one', 'two'):
        e0 = E[(key, STATES[0][0])].e_scf
        print('      primitives + %-4s diffuse: EA(7p1/2) %+.4f  EA(6d) %+.4f  converged %s' % (key, (e0 - E[(key, STATES[1][0])].e_scf) * EV,
              (e0 - E[(key, STATES[2][0])].e_scf) * EV, all(E[(key, s)].converged for s, _ in STATES)), flush=True)
    for lab, o in STATES:
        rows = contract.contraction_error(aug, 'U', 'x2camf-g', gau_nuc=True, result=E[('one', lab)])
        worst = max(rows, key=lambda r: abs(r[4] - r[3]))
        top = [r for r in rows if r[0] == max(q[0] for q in rows if q[1] == r[1]) and r[1] in (1, 2)]
        print('   %-22s contracted vs uncontracted (same primitives): largest spinor-energy error %.1e Eh (%d%s%d/2); %s' % (
            lab, abs(worst[4] - worst[3]), worst[0], LN[worst[1]], worst[2],
            '  '.join('%d%s%d/2 %.5f/%.5f' % (n, LN[l], tj, a, b) for n, l, tj, a, b in top)), flush=True)
for name, d in py.items():
    genbas.write_pyscf(os.path.join(HERE, 'pyscf', name.lower().replace('-', '_') + '.py'), d,
                       header='%s: U:DYVnZ-SOG plus one even-tempered diffuse primitive per l (draft of 2026-10-14)' % name)
print('written:', out)

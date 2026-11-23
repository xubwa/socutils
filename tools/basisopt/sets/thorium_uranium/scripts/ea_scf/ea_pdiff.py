"""Atomic SCF check of the U basis sets for the EA states, occupation fixed by spherical symmetry
(socutils SymmSpinorSCF, symmetry='sph': one block per (l, j, m_j); X2CAMF).  With these blocks
7p1/2 and 6d3/2 cannot be confused, so only electron counts per block are given.

  U    [Rn] 7s2 5f5/2(m 5/2, 3/2, 1/2) 6d3/2(m 3/2)          M_J = 6
  U-1  U + 7p1/2(m 1/2)          U-2  U + 6d3/2(m 1/2)
Checks printed: per state, the open-shell spinors that are occupied (block, energy, weight of the
block's l in the spinor); a closed-shell U6+ SCF in each basis (contracted must not lie below
uncontracted).  Results go to ea_sph_<z>.res (socutils prints a lot to stdout).

    python ea_sph.py D
"""
import sys, re
import numpy as np
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.tools.basisopt import genbas
z = sys.argv[1]
G = '/home/xwang405/scr4/zmr_runs/basis_opt/thuof/'
EV = 27.211386
OUT = open('ea_pdiff_%s.res' % z, 'w')
def say(*a):
    OUT.write(' '.join(str(x) for x in a) + '\n'); OUT.flush()
CLOSED = {'s1/2': 7, 'p1/2': 5, 'p3/2': 5, 'd3/2': 3, 'd5/2': 3, 'f5/2': 1, 'f7/2': 1}
CLOSED_6 = {'s1/2': 6, 'p1/2': 5, 'p3/2': 5, 'd3/2': 3, 'd5/2': 3, 'f5/2': 1, 'f7/2': 1}     # U6+ = [Rn]
OPEN = {('f5/2', '5/2'): 1, ('f5/2', '3/2'): 1, ('f5/2', '1/2'): 1, ('d3/2', '3/2'): 1}
STATES = [('U', 0, CLOSED, OPEN), ('U-1 (7p1/2)', -1, CLOSED, {**OPEN, ('p1/2', '1/2'): 1}),
          ('U-2 (6d3/2)', -1, CLOSED, {**OPEN, ('d3/2', '1/2'): 1}), ('U6+', 6, CLOSED_6, {})]
def occup(irreps, closed, extra):
    occ, opened = {}, []
    for key in irreps:
        m = re.match(r'([spdfghi])(\d+/2),(-?\d+/2)', key)
        lj, mj = m.group(1) + m.group(2), m.group(3)
        n = closed.get(lj, 0) + extra.get((lj, mj), 0)
        if n:
            occ[key] = [n]
        if extra.get((lj, mj), 0):
            opened.append(key)
    return occ, opened
def run(basis, charge, closed, extra, dm0=None, tagname=''):
    mol = gto.M(atom='U 0 0 0', basis={'U': basis}, charge=charge, spin=(92 - charge) % 2, verbose=0, nucmod='G')
    mf = spinor_hf.SymmSpinorSCF(mol, symmetry='sph').x2camf()
    mf.occupation, opened = occup(mf.irrep_ao, closed, extra)
    assert sum(v[0] for v in mf.occupation.values()) == mol.nelectron
    # The (l, j, m_j) blocking is a constraint, not a symmetry of a determinant with M_J != 0: the
    # two-electron potential of such a determinant couples different (l, j) of the same m_j.  Those
    # couplings are removed from the potential, so the Fock matrix, the orbital gradient and the
    # DIIS error are all block diagonal and the constrained SCF has a proper stationary point; the
    # energy is unchanged because the constrained density is block diagonal itself.
    lab = np.empty(mol.nao_2c(), dtype=object)
    for key, idx in mf.irrep_ao.items():
        lab[idx] = key
    mask = (lab[:, None] == lab[None, :])
    veff0 = mf.get_veff
    mf.get_veff = lambda *a, **k: veff0(*a, **k) * mask
    mf.conv_tol = 1e-9; mf.max_cycle = 120
    log = open('log_%s_%d_%s.log' % (tagname, charge, 'x'.join('%s%s' % kv for kv in sorted(extra.items())).replace('/', '')[:60]), 'w')
    mf.verbose = 4; mf.stdout = log; mol.stdout = log
    e = mf.kernel(dm0) if dm0 is not None else mf.kernel()
    # open-shell spinors actually occupied: the highest occupied spinor of each opened block
    info = []
    tags = np.asarray(mf.irrep_mo)
    occd = abs(mf.mo_occ) > 0.5
    s = mf.get_ovlp()
    for key in opened:
        idx = np.where((tags == key) & occd)[0]
        if len(idx) == 0:
            info.append('%s: none occupied!' % key); continue
        i = idx[np.argmax(mf.mo_energy[idx].real)]
        c = mf.mo_coeff[:, i]
        w = (c.conj() * (s @ c)).real
        lw = w[mf.irrep_ao[key]].sum() / w.sum()
        info.append('%s %+.4f (block weight %.3f, %d occupied)' % (key, mf.mo_energy[i].real, lw, len(idx)))
    return e, mf.converged, mol.nao_2c(), info, mf.make_rdm1()

import numpy as np
base = genbas.read(G + 'GENBAS.thuof_aug', 'U:ADYVTZ-SOG')
def with_p(ent, new_p, name):
    """replace the diffuse p of the entry (its last primitive / last column) by the exponents new_p"""
    shells = dict(ent.shells)
    ex, c = shells[1]
    ex = np.concatenate([ex[:-1], new_p])
    c = c[:-1, :-1]
    k = len(new_p)
    c = np.hstack([np.vstack([c, np.zeros((k, c.shape[1]))]), np.vstack([np.zeros((c.shape[0], k)), np.eye(k)])])
    shells[1] = (ex, c)
    return genbas.Entry('U:' + name, ent.comment, shells)
VARIANTS = [('ADYVTZ p 0.00667', base),
            ('ADYVTZ p 0.00521 (Lan)', with_p(base, [0.0052096], 'X1')),
            ('ADYVTZ p 0.00667+0.00285', with_p(base, [0.0066709, 0.0066709 ** 2 / 0.015629], 'X2'))]
for name, ent in VARIANTS:
    con = genbas.to_pyscf(ent, kappa=True)
    unc = [[l, [e, 1.0]] for l, (ex, c) in sorted(ent.shells.items()) for e in sorted(ex, reverse=True)]
    res = {}
    for lab, b in (('contracted', con), ('uncontracted', unc)):
        dm = None
        for st, q, closed, extra in STATES[:2]:
            e, conv, n, info, d = run(b, q, closed, extra, dm if st.startswith('U-') else None, tagname='pdiff_%s_%s' % (name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'p'), lab))
            if st == 'U': dm = d
            res[(lab, st)] = e
            say('%-26s %-26s %-26s spinors %4d  E %.8f  converged %s  %s' % (name, lab, st, n, e, conv, ' | '.join(info)))
        e0 = res[(lab, 'U')]
        say('%-26s %-26s EA_SCF: 7p1/2 %+.4f eV' % (name, lab, (e0 - res[(lab, STATES[1][0])]) * EV))
    for st, *_ in STATES[:2]:
        say('%-26s contraction error %-26s %+.3f mEh' % (name, st, (res[('contracted', st)] - res[('uncontracted', st)]) * 1e3))
    for st, *_ in STATES[1:2]:
        d = ((res[('contracted', 'U')] - res[('contracted', st)]) - (res[('uncontracted', 'U')] - res[('uncontracted', st)])) * EV * 1e3
        say('%-26s contraction error of EA %-26s %+.2f meV' % (name, st, d))

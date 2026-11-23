"""Atomic SCF check of the U basis sets for the electron-affinity states, occupation fixed by
linear symmetry (blocks of m_j; socutils SymmSpinorSCF, symmetry='linear', X2CAMF), the same
kind of single determinant as the zccsdt CFOUR runs (%dg2lin).

Electrons per m_j for U, M_J = 6 ([Rn] 7s2 5f5/2(m 5/2, 3/2, 1/2) 6d3/2(m 3/2)):
+1/2 26, -1/2 25, +3/2 15, -3/2 13, +5/2 6, -5/2 5, +-7/2 1 - the same counts as the first two
columns of the zccsdt %dg2lin for U.  Both anion states add one electron with m_j = +1/2; which
orbital it goes to is fixed with socutils' extended format [n_lowest, n_s, n_p, n_d, n_f]: the n
lowest spinors of the block, then that many further spinors of dominant s / p / d / f character.
  m_j = +1/2:  U [24, 1, 0, 0, 1] (7s, 5f)   U-1 [24, 1, 1, 0, 1] (+7p1/2)   U-2 [24, 1, 0, 1, 1] (+6d)
For each basis: the entry as made and its primitives uncontracted; SCF energies, EA_SCF =
E(U) - E(U-), the energy of the highest occupied spinor, and contracted minus uncontracted.

    python ea_scf.py TZ
"""
import sys
import numpy as np
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.tools.basisopt import genbas
z = sys.argv[1]
G = '/home/xwang405/scr4/zmr_runs/basis_opt/thuof/'
EV = 27.211386
BASE = {'-1/2': [25], '3/2': [13, 0, 0, 1, 1], '-3/2': [13], '5/2': [5, 0, 0, 0, 1], '-5/2': [5], '7/2': [1], '-7/2': [1]}
STATES = [('U', 0, {**BASE, '1/2': [24, 1, 0, 0, 1]}),
          ('U-1 (7p1/2)', -1, {**BASE, '1/2': [24, 1, 1, 0, 1]}),
          ('U-2 (6d)', -1, {**BASE, '1/2': [24, 1, 0, 1, 1]})]
def run(basis, charge, occ, dm0=None):
    mol = gto.M(atom='U 0 0 0', basis={'U': basis}, charge=charge, spin=(92 - charge) % 2, verbose=0, nucmod='G')
    assert sum(sum(v) for v in occ.values()) == mol.nelectron
    mf = spinor_hf.SymmSpinorSCF(mol, symmetry='linear', occup=occ).x2camf()
    mf.conv_tol = 1e-9; mf.max_cycle = 600; mf.damp = 0.5; mf.level_shift = 0.5; mf.diis_space = 12
    e = mf.kernel(dm0) if dm0 is not None else mf.kernel()
    if not mf.converged:                                  # second pass from where it stopped, no shift
        mf.level_shift = 0.0; mf.damp = 0.0
        e = mf.kernel(mf.make_rdm1())
    o = np.where(abs(mf.mo_occ) > 0.5)[0]
    return e, mf.converged, max(mf.mo_energy[o].real), mol.nao_2c(), mf.make_rdm1()
for name, src in (('ADYV%sZ-SOG' % z, 'GENBAS.thuof_aug'), ('DYV%sZ-SOG' % z, 'GENBAS.thuof')):
    ent = genbas.read(G + src, 'U:' + name)
    con = genbas.to_pyscf(ent, kappa=True)
    unc = [[l, [e, 1.0]] for l, (ex, c) in sorted(ent.shells.items()) for e in sorted(ex, reverse=True)]
    res = {}
    for lab, b in (('contracted', con), ('uncontracted', unc)):
        dm = None
        for st, q, occ in STATES:
            e, conv, eh, n, d = run(b, q, occ, dm)
            if st == 'U': dm = d                          # anions start from the neutral density
            res[(lab, st)] = e
            print('%-12s %-12s %-12s spinors %4d  E %.8f  converged %s  highest occupied spinor %+.5f Eh' % (name, lab, st, n, e, conv, eh), flush=True)
        e0 = res[(lab, 'U')]
        print('%-12s %-12s EA_SCF: 7p1/2 %+.4f eV   6d %+.4f eV' % (name, lab, (e0 - res[(lab, STATES[1][0])]) * EV, (e0 - res[(lab, STATES[2][0])]) * EV), flush=True)
    for st, _, _ in STATES:
        print('%-12s contraction error %-12s %+.3f mEh' % (name, st, (res[('contracted', st)] - res[('uncontracted', st)]) * 1e3))
    for st, _, _ in STATES[1:]:
        d = ((res[('contracted', 'U')] - res[('contracted', st)]) - (res[('uncontracted', 'U')] - res[('uncontracted', st)])) * EV * 1e3
        print('%-12s contraction error of EA %-12s %+.2f meV' % (name, st, d), flush=True)

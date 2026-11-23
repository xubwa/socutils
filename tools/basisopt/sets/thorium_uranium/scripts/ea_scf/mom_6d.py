"""U- 6d state with maximum-overlap occupation (the QZ SCF of ea_sph.py oscillated: inside the
d3/2,m=1/2 block the 6d and a diffuse d of similar energy swap order, so "lowest 4" flips).

Masked (l, j, m_j)-constrained SCF as in ea_sph.py.  Neutral U (M_J = 6) by aufbau inside each
block; then U- 6d: start from the neutral density, reference occupied spinors = the neutral ones
plus the lowest unoccupied spinor of block d3/2,1/2 of the neutral (its 6d); at every iteration
the occupied spinors of a block are those with the largest projection on the previous occupied
spinors of that block.  Contracted and uncontracted, TZ (check against ea_sph_T.res) and QZ.

    python mom_6d.py Q
"""
import re, sys
import numpy as np
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.tools.basisopt import genbas
z = sys.argv[1]
G = '/home/xwang405/scr4/zmr_runs/basis_opt/thuof/'
EV = 27.211386
OUT = open('mom_6d_%s.res' % z, 'w')
def say(*a):
    OUT.write(' '.join(str(x) for x in a) + '\n'); OUT.flush()
CLOSED = {'s1/2': 7, 'p1/2': 5, 'p3/2': 5, 'd3/2': 3, 'd5/2': 3, 'f5/2': 1, 'f7/2': 1}
OPEN = {('f5/2', '5/2'): 1, ('f5/2', '3/2'): 1, ('f5/2', '1/2'): 1, ('d3/2', '3/2'): 1}
def setup(basis, charge, extra, logname):
    mol = gto.M(atom='U 0 0 0', basis={'U': basis}, charge=charge, spin=(92 - charge) % 2, verbose=0, nucmod='G', max_memory=int(__import__('os').environ.get('MAXMEM', '100000')))
    mf = spinor_hf.SymmSpinorSCF(mol, symmetry='sph').x2camf()
    occ = {}
    for key in mf.irrep_ao:
        m = re.match(r'([spdfghi])(\d+/2),(-?\d+/2)', key)
        n = CLOSED.get(m.group(1) + m.group(2), 0) + extra.get((m.group(1) + m.group(2), m.group(3)), 0)
        if n: occ[key] = [n]
    mf.occupation = occ
    lab = np.empty(mol.nao_2c(), dtype=object)
    for key, idx in mf.irrep_ao.items():
        lab[idx] = key
    mask = (lab[:, None] == lab[None, :])
    veff0 = mf.get_veff
    mf.get_veff = lambda *a, **k: veff0(*a, **k) * mask
    mf.conv_tol = 1e-9; mf.max_cycle = 150
    mf.verbose = 4; mf.stdout = open(logname, 'w')
    return mol, mf
def mom(mf, ref_coeff, nblock):
    """occupation by maximum overlap with the previous occupied spinors of each block"""
    s = mf.get_ovlp()
    state = {'prev': ref_coeff}
    def get_occ(mo_energy=None, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        tags = np.asarray(mf.irrep_mo)
        occ = np.zeros(len(mo_energy))
        new_prev = {}
        for key, n in nblock.items():
            idx = np.where(tags == key)[0]
            prev = state['prev'][key]
            p = np.sum(abs(prev.conj().T @ s @ mo_coeff[:, idx]) ** 2, axis=0)
            pick = idx[np.argsort(-p)[:n]]
            occ[pick] = 1.0
            new_prev[key] = mo_coeff[:, pick]
        state['prev'] = new_prev
        return occ
    mf.get_occ = get_occ
    return mf
def info(mf, keys):
    tags = np.asarray(mf.irrep_mo); o = abs(mf.mo_occ) > 0.5
    out = []
    for k in keys:
        idx = np.where((tags == k) & o)[0]
        out.append('%s %+.4f' % (k, mf.mo_energy[idx].real.max()))
    return ' | '.join(out)
for name, src in (('ADYV%sZ-SOG' % z, 'GENBAS.thuof_aug'),):
    ent = genbas.read(G + src, 'U:' + name)
    res = {}
    for lab, b in (('contracted', genbas.to_pyscf(ent, kappa=True)),
                   ('uncontracted', [[l, [e, 1.0]] for l, (ex, c) in sorted(ent.shells.items()) for e in sorted(ex, reverse=True)])):
        mol0, mf0 = setup(b, 0, OPEN, 'mom_%s_%s_U.log' % (name, lab))
        e0 = mf0.kernel()
        say('%-12s %-12s U           E %.8f converged %s  %s' % (name, lab, e0, mf0.converged, info(mf0, ['d3/2,3/2', 'f5/2,1/2'])))
        # reference for U-: neutral occupied spinors per block, plus the lowest unoccupied of d3/2,1/2
        tags = np.asarray(mf0.irrep_mo); o = abs(mf0.mo_occ) > 0.5
        ref, nblock = {}, {}
        for key in mf0.irrep_ao:
            idx = np.where(tags == key)[0]
            occ_idx = idx[o[idx]]
            if key == 'd3/2,1/2':
                virt = idx[~o[idx]]
                add = virt[np.argmin(mf0.mo_energy[virt].real)]
                say('    6d m1/2 taken from the neutral: spinor energy %+.4f' % mf0.mo_energy[add].real)
                occ_idx = np.concatenate([occ_idx, [add]])
            if len(occ_idx):
                ref[key] = mf0.mo_coeff[:, occ_idx]; nblock[key] = len(occ_idx)
        mol1, mf1 = setup(b, -1, {**OPEN, ('d3/2', '1/2'): 1}, 'mom_%s_%s_Um2.log' % (name, lab))
        mf1 = mom(mf1, ref, nblock)
        e1 = mf1.kernel(mf0.make_rdm1())
        say('%-12s %-12s U-2 (6d)    E %.8f converged %s  %s' % (name, lab, e1, mf1.converged, info(mf1, ['d3/2,1/2', 'd3/2,3/2', 'f5/2,1/2'])))
        res[lab] = (e0, e1)
        say('%-12s %-12s EA_SCF(6d) %+.4f eV' % (name, lab, (e0 - e1) * EV))
    d = ((res['contracted'][0] - res['contracted'][1]) - (res['uncontracted'][0] - res['uncontracted'][1])) * EV * 1e3
    say('%-12s contraction error of EA(6d) %+.2f meV' % (name, d))
